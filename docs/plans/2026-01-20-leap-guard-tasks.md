# LEAP-Guard 360 Task Breakdown

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a predictive maintenance dashboard with LSTM anomaly detection and GenAI diagnostics

**Architecture:** React frontend → Lambda (containerized) → LSTM model + Bedrock GenAI

**Tech Stack:** Python 3.12, PyTorch, Pydantic, AWS SAM, React, TypeScript, Recharts

---

## Phase 1: Data & Model (Google Colab)

> **Note:** Phase 1 is executed in Google Colab, not locally. These are reference steps.

### Task 1.1: Colab Environment Setup

**Step 1:** Create new Colab notebook named `leap_guard_training.ipynb`

**Step 2:** Install dependencies
```python
!pip install torch pandas scikit-learn matplotlib
```

**Step 3:** Verify GPU available
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
# Expected: CUDA available: True, GPU: Tesla T4
```

---

### Task 1.2: Download CMAPSS Dataset

**Step 1:** Download FD001 subset
```python
!wget -q "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip" -O CMAPSSData.zip
!unzip -q CMAPSSData.zip -d data/
```

**Step 2:** Load and inspect data
```python
import pandas as pd

columns = ['unit', 'cycle'] + [f'op{i}' for i in range(1,4)] + [f's{i}' for i in range(1,22)]
train_df = pd.read_csv('data/train_FD001.txt', sep=' ', header=None, names=columns)
train_df = train_df.dropna(axis=1)  # Drop trailing NaN columns
print(f"Shape: {train_df.shape}")  # Expected: (20631, 26)
print(f"Engines: {train_df['unit'].nunique()}")  # Expected: 100
```

---

### Task 1.3: Preprocess Data

**Step 1:** Check variance and drop constant sensors
```python
# Check variance of all sensor columns
sensor_cols = [c for c in train_df.columns if c.startswith('s') or c.startswith('op')]
variances = train_df[sensor_cols].var().sort_values()
print("Variance per column (lowest first):")
print(variances)

# Drop columns with variance < 0.0001 (effectively constant)
low_var_cols = variances[variances < 0.0001].index.tolist()
print(f"\nDropping low-variance columns: {low_var_cols}")

# Keep only sensor columns with meaningful variance
feature_cols = [c for c in train_df.columns if c.startswith('s') and c not in low_var_cols]
print(f"Features to use: {len(feature_cols)} -> {feature_cols}")
```

**Step 2:** Normalize with MinMaxScaler
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
```

**Step 3:** Create sliding windows (healthy cycles only)
```python
import numpy as np

def create_sequences(df, seq_length=50, healthy_pct=0.2):
    sequences = []
    for unit in df['unit'].unique():
        unit_data = df[df['unit'] == unit][feature_cols].values
        max_healthy = int(len(unit_data) * healthy_pct)
        for i in range(max_healthy - seq_length):
            sequences.append(unit_data[i:i+seq_length])
    return np.array(sequences)

X_train = create_sequences(train_df)
print(f"Training sequences: {X_train.shape}")  # Expected: (626, 50, 13)
```

**Step 4:** Split by engine unit (random, reproducible)
```python
from sklearn.model_selection import train_test_split

# Split engine units 80/20 (not rows - prevents data leakage)
all_units = train_df['unit'].unique()
train_units, val_units = train_test_split(all_units, test_size=0.2, random_state=42)

print(f"Train engines: {len(train_units)}, Val engines: {len(val_units)}")

X_train = create_sequences(train_df[train_df['unit'].isin(train_units)])
X_val = create_sequences(train_df[train_df['unit'].isin(val_units)])
print(f"Train: {X_train.shape}, Val: {X_val.shape}")
```

---

### Task 1.4: Build LSTM-Autoencoder

**Step 1:** Define model architecture
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features=14, hidden_dim=64, latent_dim=32, seq_len=50):
        super().__init__()
        self.seq_len = seq_len
        self.encoder_lstm1 = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.encoder_lstm2 = nn.LSTM(hidden_dim, latent_dim, batch_first=True)
        self.decoder_lstm1 = nn.LSTM(latent_dim, latent_dim, batch_first=True)
        self.decoder_lstm2 = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        x, _ = self.encoder_lstm1(x)
        x, (hidden, _) = self.encoder_lstm2(x)
        latent = hidden.squeeze(0).unsqueeze(1).repeat(1, self.seq_len, 1)
        x, _ = self.decoder_lstm1(latent)
        x, _ = self.decoder_lstm2(x)
        return self.output_layer(x)

model = LSTMAutoencoder()
print(model)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**Step 2:** Train with early stopping
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

train_tensor = torch.FloatTensor(X_train).to(device)
val_tensor = torch.FloatTensor(X_val).to(device)
train_loader = DataLoader(TensorDataset(train_tensor, train_tensor), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(val_tensor, val_tensor), batch_size=32)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_val_loss = float('inf')
patience, patience_counter = 5, 0
train_losses, val_losses = [], []

for epoch in range(50):
    model.train()
    train_loss = sum(criterion(model(X), X).item() for X, _ in train_loader) / len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    with torch.no_grad():
        val_loss = sum(criterion(model(X), X).item() for X, _ in val_loader) / len(val_loader)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}: Train={train_loss:.6f}, Val={val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

model.load_state_dict(best_state)
```

**Step 3:** Plot training history
```python
import matplotlib.pyplot as plt

plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Val')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.savefig('training_history.png')
```

---

### Task 1.5: Compute Anomaly Threshold

**Step 1:** Calculate reconstruction errors on healthy validation data
```python
model.eval()
with torch.no_grad():
    X_val_pred = model(val_tensor).cpu().numpy()
X_val_np = val_tensor.cpu().numpy()
mse = np.mean(np.power(X_val_np - X_val_pred, 2), axis=(1, 2))
threshold = np.percentile(mse, 95)
print(f"Threshold (95th percentile): {threshold:.6f}")
```

**Step 2:** Visualize error distribution
```python
plt.figure(figsize=(10, 4))
plt.hist(mse, bins=50, alpha=0.7)
plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')
plt.xlabel('Reconstruction Error (MSE)')
plt.ylabel('Count')
plt.legend()
plt.savefig('error_distribution.png')
```

---

### Task 1.6: Export Model Artifacts

**Step 1:** Save model
```python
import os
torch.save(model.cpu().state_dict(), 'leap_guard_model.pth')
print(f"Model size: {os.path.getsize('leap_guard_model.pth') / 1e6:.1f} MB")
```

**Step 2:** Save scaler
```python
import pickle
with open('leap_guard_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
```

**Step 3:** Save threshold
```python
import json
with open('threshold.json', 'w') as f:
    json.dump({"threshold": float(threshold)}, f)
```

**Step 4:** Save config
```python
config = {
    "features": feature_cols,
    "window_size": 50,
    "n_features": 14
}
with open('config.json', 'w') as f:
    json.dump(config, f, indent=2)
```

**Step 5:** Download all artifacts
```python
from google.colab import files
files.download('leap_guard_model.pth')
files.download('leap_guard_scaler.pkl')
files.download('threshold.json')
files.download('config.json')
```

---

## Phase 2: Backend & Cloud

### Task 2.1: Create Backend Directory Structure

**Files:**
- Create: `backend/app/__init__.py`
- Create: `backend/app/schemas.py`
- Create: `backend/requirements.txt`

**Step 1:** Create directory structure

```bash
mkdir -p backend/app backend/model backend/tests backend/events
```

Run: `mkdir -p backend/app backend/model backend/tests backend/events`
Expected: Directories created

**Step 2:** Create empty `__init__.py`

```python
# backend/app/__init__.py
# Package marker
```

**Step 3:** Create `requirements.txt`

```
torch>=2.2.0
numpy>=1.26.0
pydantic>=2.5.0
boto3>=1.34.0
scikit-learn>=1.4.0
pytest>=8.0.0
```

**Step 4:** Commit

```bash
git add backend/
git commit -m "chore: create backend directory structure"
```

---

### Task 2.2: Implement Pydantic Schemas

**Files:**
- Create: `backend/app/schemas.py`
- Create: `backend/tests/test_schemas.py`

**Step 1:** Write the failing test

```python
# backend/tests/test_schemas.py
import pytest
from app.schemas import PredictRequest, PredictResponse

def test_predict_request_valid():
    request = PredictRequest(
        sensor_readings=[[1.0] * 14] * 50,
        window_size=50,
        threshold=0.7
    )
    assert request.window_size == 50
    assert request.threshold == 0.7

def test_predict_request_defaults():
    request = PredictRequest(
        sensor_readings=[[1.0] * 14] * 50
    )
    assert request.window_size == 50
    assert request.threshold is None

def test_predict_request_invalid_window():
    with pytest.raises(ValueError):
        PredictRequest(
            sensor_readings=[[1.0] * 14] * 50,
            window_size=5  # Below minimum of 10
        )

def test_predict_request_empty_readings():
    with pytest.raises(ValueError):
        PredictRequest(
            sensor_readings=[]
        )

def test_predict_response():
    response = PredictResponse(
        anomaly_score=0.85,
        threshold=0.7,
        is_anomaly=True,
        diagnosis="Test diagnosis",
        sensor_contributions={"T30": 0.4, "P30": 0.3}
    )
    assert response.is_anomaly is True
```

**Step 2:** Run test to verify it fails

Run: `cd backend && python -m pytest tests/test_schemas.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'app.schemas'"

**Step 3:** Write minimal implementation

```python
# backend/app/schemas.py
from pydantic import BaseModel, Field, field_validator
from typing import Optional

class PredictRequest(BaseModel):
    sensor_readings: list[list[float]]
    window_size: int = Field(default=50, ge=10, le=100)
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator("sensor_readings")
    @classmethod
    def validate_readings(cls, v):
        if not v or not v[0]:
            raise ValueError("sensor_readings cannot be empty")
        return v

class PredictResponse(BaseModel):
    anomaly_score: float
    threshold: float
    is_anomaly: bool
    diagnosis: Optional[str] = None
    sensor_contributions: dict[str, float]
```

**Step 4:** Run test to verify it passes

Run: `cd backend && python -m pytest tests/test_schemas.py -v`
Expected: PASS (5 passed)

**Step 5:** Commit

```bash
git add backend/app/schemas.py backend/tests/test_schemas.py
git commit -m "feat: add Pydantic request/response schemas with validation"
```

---

### Task 2.3: Implement Inference Module

**Files:**
- Create: `backend/app/inference.py`
- Create: `backend/tests/test_inference.py`
- Requires: Model artifacts in `backend/model/`

**Step 1:** Copy model artifacts to backend/model/

```bash
cp ~/Downloads/leap_guard_model.pth backend/model/
cp ~/Downloads/leap_guard_scaler.pkl backend/model/
cp ~/Downloads/threshold.json backend/model/
cp ~/Downloads/config.json backend/model/
```

**Step 2:** Write the failing test

```python
# backend/tests/test_inference.py
import pytest
import numpy as np
from app.inference import AnomalyDetector

@pytest.fixture
def detector():
    return AnomalyDetector(model_dir="model/")

def test_detector_loads(detector):
    assert detector.model is not None
    assert detector.scaler is not None
    assert detector.default_threshold > 0

def test_predict_healthy_data(detector):
    # Generate healthy-looking data (normalized, low variance)
    readings = np.random.uniform(0.3, 0.7, (50, 14)).tolist()
    result = detector.predict(readings, window_size=50)

    assert "anomaly_score" in result
    assert "is_anomaly" in result
    assert "sensor_contributions" in result
    assert isinstance(result["anomaly_score"], float)

def test_predict_returns_top_3_contributors(detector):
    readings = np.random.uniform(0.3, 0.7, (50, 14)).tolist()
    result = detector.predict(readings, window_size=50)

    assert len(result["sensor_contributions"]) == 3

def test_predict_invalid_shape(detector):
    readings = [[1.0] * 10] * 30  # Wrong shape
    with pytest.raises(ValueError, match="Expected shape"):
        detector.predict(readings, window_size=50)
```

**Step 3:** Run test to verify it fails

Run: `cd backend && python -m pytest tests/test_inference.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'app.inference'"

**Step 4:** Write implementation

```python
# backend/app/inference.py
import numpy as np
import pickle
import json
import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    """Must match the architecture used during training."""
    def __init__(self, n_features=14, hidden_dim=64, latent_dim=32, seq_len=50):
        super().__init__()
        self.seq_len = seq_len
        self.encoder_lstm1 = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.encoder_lstm2 = nn.LSTM(hidden_dim, latent_dim, batch_first=True)
        self.decoder_lstm1 = nn.LSTM(latent_dim, latent_dim, batch_first=True)
        self.decoder_lstm2 = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        x, _ = self.encoder_lstm1(x)
        x, (hidden, _) = self.encoder_lstm2(x)
        latent = hidden.squeeze(0).unsqueeze(1).repeat(1, self.seq_len, 1)
        x, _ = self.decoder_lstm1(latent)
        x, _ = self.decoder_lstm2(x)
        return self.output_layer(x)

class AnomalyDetector:
    def __init__(self, model_dir: str = "model/"):
        with open(f"{model_dir}/config.json", "r") as f:
            self.config = json.load(f)

        self.model = LSTMAutoencoder(
            n_features=self.config["n_features"],
            seq_len=self.config["window_size"]
        )
        self.model.load_state_dict(torch.load(f"{model_dir}/leap_guard_model.pth", map_location="cpu"))
        self.model.eval()

        with open(f"{model_dir}/leap_guard_scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        with open(f"{model_dir}/threshold.json", "r") as f:
            self.default_threshold = json.load(f)["threshold"]

    def predict(self, sensor_readings: list, window_size: int, threshold: float = None):
        threshold = threshold or self.default_threshold

        X = np.array(sensor_readings)
        expected_shape = (window_size, self.config["n_features"])
        if X.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, got {X.shape}")

        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0)

        with torch.no_grad():
            X_reconstructed = self.model(X_tensor).numpy()

        mse_per_feature = np.mean((X_scaled - X_reconstructed[0]) ** 2, axis=0)
        anomaly_score = float(np.mean(mse_per_feature))

        total_error = np.sum(mse_per_feature)
        top_indices = np.argsort(mse_per_feature)[-3:][::-1]
        contributions = {
            self.config["features"][i]: round(float(mse_per_feature[i] / total_error), 3)
            for i in top_indices
        }

        return {
            "anomaly_score": round(anomaly_score, 4),
            "is_anomaly": anomaly_score > threshold,
            "sensor_contributions": contributions
        }
```

**Step 5:** Run test to verify it passes

Run: `cd backend && python -m pytest tests/test_inference.py -v`
Expected: PASS (4 passed)

**Step 6:** Commit

```bash
git add backend/app/inference.py backend/tests/test_inference.py backend/model/
git commit -m "feat: add AnomalyDetector with LSTM inference"
```

---

### Task 2.4: Implement Bedrock Client

**Files:**
- Create: `backend/app/bedrock_client.py`
- Create: `backend/tests/test_bedrock_client.py`

**Step 1:** Write the failing test

```python
# backend/tests/test_bedrock_client.py
import pytest
import os
from app.bedrock_client import BedrockDiagnostics

@pytest.fixture
def mock_client(monkeypatch):
    monkeypatch.setenv("MOCK_BEDROCK", "true")
    return BedrockDiagnostics()

def test_mock_mode_enabled(mock_client):
    assert mock_client.mock_mode is True

def test_diagnose_returns_string(mock_client):
    result = mock_client.diagnose(
        sensor_values=[0.5] * 14,
        anomaly_score=0.85,
        top_contributors={"T30": 0.4, "P30": 0.3, "Nc": 0.2}
    )
    assert isinstance(result, str)
    assert "T30" in result  # Top contributor mentioned

def test_sensor_descriptions_exist(mock_client):
    assert "T30" in mock_client.sensor_descriptions
    assert "P30" in mock_client.sensor_descriptions
```

**Step 2:** Run test to verify it fails

Run: `cd backend && python -m pytest tests/test_bedrock_client.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3:** Write implementation

```python
# backend/app/bedrock_client.py
import boto3
import json
import os
from botocore.config import Config

class BedrockDiagnostics:
    def __init__(self):
        config = Config(read_timeout=10, retries={"max_attempts": 2})

        self.client = boto3.client(
            "bedrock-runtime",
            region_name="ap-southeast-1",
            config=config
        )

        self.model_id = "anthropic.claude-3-haiku-20240307-v1:0"
        self.mock_mode = os.environ.get("MOCK_BEDROCK", "false").lower() == "true"

        self.sensor_descriptions = {
            "T24": "Low Pressure Compressor outlet temperature",
            "T30": "High Pressure Compressor outlet temperature",
            "T50": "Low Pressure Turbine outlet temperature",
            "P30": "High Pressure Compressor outlet pressure",
            "Ps30": "Static pressure at HPC outlet",
            "Nf": "Fan rotational speed (RPM)",
            "Nc": "Core rotational speed (RPM)",
            "phi": "Flow coefficient",
            "BPR": "Bypass ratio",
            "htBleed": "Bleed enthalpy",
            "s2": "Sensor 2",
            "s3": "Sensor 3",
            "s4": "Sensor 4",
            "s7": "Sensor 7",
            "s8": "Sensor 8",
            "s9": "Sensor 9",
            "s11": "Sensor 11",
            "s12": "Sensor 12",
            "s13": "Sensor 13",
            "s14": "Sensor 14",
            "s15": "Sensor 15",
            "s17": "Sensor 17",
            "s20": "Sensor 20",
            "s21": "Sensor 21",
        }

    def diagnose(
        self,
        sensor_values: list[float],
        anomaly_score: float,
        top_contributors: dict[str, float]
    ) -> str:
        if self.mock_mode:
            return self._mock_response(top_contributors)

        sensor_context = "\n".join([
            f"- {sensor} ({self.sensor_descriptions.get(sensor, 'Unknown')}): "
            f"contributing {pct*100:.1f}% to anomaly"
            for sensor, pct in top_contributors.items()
        ])

        prompt = f"""You are an aircraft engine diagnostic expert specializing in CFM LEAP turbofan engines.

An anomaly has been detected in engine sensor data with a score of {anomaly_score:.3f}.

The sensors contributing most to this anomaly are:
{sensor_context}

Based on these sensor patterns, provide:
1. A likely root cause (1-2 sentences)
2. The affected engine component/system
3. Recommended maintenance action

Keep your response under 100 words. Use technical but accessible language suitable for an MRO engineer."""

        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 200,
                    "temperature": 0.3,
                    "messages": [{"role": "user", "content": prompt}]
                })
            )
            result = json.loads(response["body"].read())
            return result["content"][0]["text"]
        except Exception as e:
            print(f"Bedrock error: {e}")
            return None

    def _mock_response(self, top_contributors: dict) -> str:
        top_sensor = list(top_contributors.keys())[0]
        return (
            f"[MOCK] Elevated {top_sensor} readings suggest possible HPC seal degradation. "
            f"Recommend borescope inspection of high-pressure compressor at next scheduled maintenance."
        )
```

**Step 4:** Run test to verify it passes

Run: `cd backend && MOCK_BEDROCK=true python -m pytest tests/test_bedrock_client.py -v`
Expected: PASS (3 passed)

**Step 5:** Commit

```bash
git add backend/app/bedrock_client.py backend/tests/test_bedrock_client.py
git commit -m "feat: add BedrockDiagnostics client with mock mode"
```

---

### Task 2.5: Implement Lambda Handler

**Files:**
- Create: `backend/app/handler.py`
- Create: `backend/tests/test_handler.py`

**Step 1:** Write the failing test

```python
# backend/tests/test_handler.py
import pytest
import json
import os

os.environ["MOCK_BEDROCK"] = "true"

from app.handler import lambda_handler

def test_handler_valid_request():
    event = {
        "body": json.dumps({
            "sensor_readings": [[0.5] * 14] * 50,
            "window_size": 50,
            "threshold": 0.7
        })
    }
    response = lambda_handler(event, None)

    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert "anomaly_score" in body
    assert "is_anomaly" in body

def test_handler_invalid_request():
    event = {"body": json.dumps({"sensor_readings": []})}
    response = lambda_handler(event, None)

    assert response["statusCode"] == 400

def test_handler_missing_body():
    event = {}
    response = lambda_handler(event, None)

    assert response["statusCode"] in [400, 500]
```

**Step 2:** Run test to verify it fails

Run: `cd backend && python -m pytest tests/test_handler.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3:** Write implementation

```python
# backend/app/handler.py
import json
from pydantic import ValidationError
from app.inference import AnomalyDetector
from app.bedrock_client import BedrockDiagnostics
from app.schemas import PredictRequest, PredictResponse

# Load outside handler for warm invocations
detector = AnomalyDetector()
diagnostics = BedrockDiagnostics()

def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))
        request = PredictRequest(**body)

        result = detector.predict(
            sensor_readings=request.sensor_readings,
            window_size=request.window_size,
            threshold=request.threshold
        )

        diagnosis = None
        if result["is_anomaly"]:
            diagnosis = diagnostics.diagnose(
                sensor_values=request.sensor_readings[-1],
                anomaly_score=result["anomaly_score"],
                top_contributors=result["sensor_contributions"]
            )

        response = PredictResponse(
            anomaly_score=result["anomaly_score"],
            threshold=request.threshold or detector.default_threshold,
            is_anomaly=result["is_anomaly"],
            diagnosis=diagnosis,
            sensor_contributions=result["sensor_contributions"]
        )

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": response.model_dump_json()
        }

    except ValidationError as e:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "Invalid input", "detail": str(e)})
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "Inference failed", "detail": str(e)})
        }
```

**Step 4:** Run test to verify it passes

Run: `cd backend && MOCK_BEDROCK=true python -m pytest tests/test_handler.py -v`
Expected: PASS (3 passed)

**Step 5:** Commit

```bash
git add backend/app/handler.py backend/tests/test_handler.py
git commit -m "feat: add Lambda handler with error handling"
```

---

### Task 2.6: Create Dockerfile

**Files:**
- Create: `backend/Dockerfile`

**Step 1:** Write Dockerfile

```dockerfile
# backend/Dockerfile
FROM public.ecr.aws/lambda/python:3.12

COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ${LAMBDA_TASK_ROOT}/app/
COPY model/ ${LAMBDA_TASK_ROOT}/model/

CMD ["app.handler.lambda_handler"]
```

**Step 2:** Build image locally

Run: `cd backend && docker build -t leap-guard-inference .`
Expected: Successfully built (may take 3-5 minutes first time)

**Step 3:** Test container locally

Run: `docker run -p 9000:8080 -e MOCK_BEDROCK=true leap-guard-inference`

In another terminal:
```bash
curl -X POST "http://localhost:9000/2015-03-31/functions/function/invocations" \
  -d '{"body": "{\"sensor_readings\": [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]] * 50, \"window_size\": 50}"}'
```
Expected: JSON response with anomaly_score

**Step 4:** Commit

```bash
git add backend/Dockerfile
git commit -m "chore: add Dockerfile for Lambda container"
```

---

### Task 2.7: Create SAM Template

**Files:**
- Create: `backend/template.yaml`
- Create: `backend/events/test_event.json`

**Step 1:** Create SAM template

```yaml
# backend/template.yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31
Description: LEAP-Guard 360 Inference API

Globals:
  Function:
    Timeout: 30
    MemorySize: 1024

Resources:
  InferenceFunction:
    Type: AWS::Serverless::Function
    Properties:
      PackageType: Image
      Architectures:
        - x86_64
      FunctionUrlConfig:
        AuthType: NONE
        Cors:
          AllowOrigins:
            - "*"
          AllowMethods:
            - POST
            - OPTIONS
          AllowHeaders:
            - Content-Type
      Policies:
        - Version: "2012-10-17"
          Statement:
            - Effect: Allow
              Action:
                - bedrock:InvokeModel
              Resource: "arn:aws:bedrock:ap-southeast-1::foundation-model/anthropic.claude-3-haiku-20240307-v1:0"
      Environment:
        Variables:
          MOCK_BEDROCK: "false"
    Metadata:
      DockerTag: latest
      DockerContext: .
      Dockerfile: Dockerfile

Outputs:
  FunctionUrl:
    Description: Lambda Function URL
    Value: !GetAtt InferenceFunctionUrl.FunctionUrl
```

**Step 2:** Create test event

```json
// backend/events/test_event.json
{
  "body": "{\"sensor_readings\": [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]], \"window_size\": 50, \"threshold\": 0.7}"
}
```

**Step 3:** Validate SAM template

Run: `cd backend && sam validate`
Expected: "template.yaml is a valid SAM Template"

**Step 4:** Commit

```bash
git add backend/template.yaml backend/events/test_event.json
git commit -m "chore: add SAM template and test event"
```

---

### Task 2.8: Deploy to AWS

**Prerequisites:**
- AWS CLI configured
- Bedrock model access approved
- Docker running

**Step 1:** Create ECR repository

```bash
aws ecr create-repository --repository-name leap-guard-inference --region ap-southeast-1
```

**Step 2:** Build with SAM

Run: `cd backend && sam build`
Expected: "Build Succeeded"

**Step 3:** Deploy with SAM (first time)

Run: `cd backend && sam deploy --guided`

Answer prompts:
- Stack Name: `leap-guard-inference`
- Region: `ap-southeast-1`
- Confirm changes: `Y`
- Allow SAM CLI IAM role creation: `Y`
- InferenceFunction may not have authorization defined, Is this okay? `Y`
- Save arguments to samconfig.toml: `Y`

**Step 4:** Get Function URL

```bash
aws cloudformation describe-stacks \
  --stack-name leap-guard-inference \
  --query "Stacks[0].Outputs[?OutputKey=='FunctionUrl'].OutputValue" \
  --output text
```

**Step 5:** Test deployed endpoint

```bash
curl -X POST <function-url> \
  -H "Content-Type: application/json" \
  -d '{"sensor_readings": [[0.5]*14]*50, "window_size": 50}'
```

**Step 6:** Commit samconfig

```bash
git add backend/samconfig.toml
git commit -m "chore: add SAM deployment config"
```

---

## Phase 3: Frontend & Integration

> **Design Reference:** See `docs/leap-guard-design.pen` for approved UI mockups

### Design System (from approved Pencil design)

**Color Tokens:**
```css
/* Light Theme (Landing Page) */
--light-bg-primary: #FFFFFF;
--light-bg-secondary: #F5F5F7;
--light-text-primary: #1D1D1F;
--light-text-secondary: #86868B;

/* Dark Theme (Dashboard) */
--bg-primary: #0D0D0D;
--bg-secondary: #1A1A1A;
--bg-tertiary: #2A2A2A;
--text-primary: #FFFFFF;
--text-secondary: #8E8E93;

/* Accent Colors */
--accent-blue: #007AFF;
--success-green: #32D583;
--warning-orange: #FF9900;
--alert-red: #E85A4F;
```

**Pages to Implement:**
1. **Landing Page** - Light theme, marketing/intro (1440px max-width)
2. **Dashboard Page** - Dark theme, main app interface (full viewport, 3-column grid)

---

### Task 3.1: Initialize React Project

**Step 1:** Create Vite project

```bash
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
```

**Step 2:** Install dependencies

```bash
npm install recharts axios lucide-react react-router-dom
npm install -D @types/node
```

**Step 3:** Verify dev server starts

Run: `npm run dev`
Expected: Server running at http://localhost:5173

**Step 4:** Commit

```bash
git add frontend/
git commit -m "chore: initialize React frontend with Vite"
```

---

### Task 3.2: Create TypeScript Types

**Files:**
- Create: `frontend/src/types/api.ts`

**Step 1:** Create types file

```typescript
// frontend/src/types/api.ts
export interface SensorReading {
  cycle: number;
  sensors: Record<string, number>;
}

export interface PredictRequest {
  sensor_readings: number[][];
  window_size: number;
  threshold: number;
}

export interface PredictResponse {
  anomaly_score: number;
  threshold: number;
  is_anomaly: boolean;
  diagnosis: string | null;
  sensor_contributions: Record<string, number>;
}

export interface EngineData {
  engine_id: string;
  metadata: {
    aircraft_type: string;
    total_cycles: number;
  };
  cycles: SensorReading[];
}

export interface AnomalyRegion {
  start: number;
  end: number;
}
```

**Step 2:** Commit

```bash
git add frontend/src/types/
git commit -m "feat: add TypeScript API types"
```

---

### Task 3.3: Create useInference Hook

**Files:**
- Create: `frontend/src/hooks/useInference.ts`

**Step 1:** Create hook

```typescript
// frontend/src/hooks/useInference.ts
import { useState, useCallback } from 'react';
import axios from 'axios';
import { PredictRequest, PredictResponse } from '../types/api';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3000';

export function useInference() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);

  const predict = useCallback(async (request: PredictRequest) => {
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post<PredictResponse>(API_URL, request);
      setResult(response.data);
      return response.data;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Inference failed';
      setError(message);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  return { predict, loading, error, result, reset };
}
```

**Step 2:** Create environment file

```
# frontend/.env.local
VITE_API_URL=http://localhost:3000
```

**Step 3:** Commit

```bash
git add frontend/src/hooks/ frontend/.env.local
git commit -m "feat: add useInference hook for API calls"
```

---

### Task 3.4: Create SensorChart Component

**Files:**
- Create: `frontend/src/components/SensorChart.tsx`

**Step 1:** Create component

```typescript
// frontend/src/components/SensorChart.tsx
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceArea,
  ResponsiveContainer,
  Legend
} from 'recharts';
import { AnomalyRegion } from '../types/api';

interface ChartData {
  cycle: number;
  actual: number;
  predicted: number;
}

interface Props {
  data: ChartData[];
  anomalyRegions: AnomalyRegion[];
  onRegionClick: (start: number, end: number) => void;
  selectedRegion: AnomalyRegion | null;
}

export function SensorChart({ data, anomalyRegions, onRegionClick, selectedRegion }: Props) {
  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
        <XAxis
          dataKey="cycle"
          stroke="#64748b"
          label={{ value: 'Cycle', position: 'bottom', fill: '#64748b' }}
        />
        <YAxis stroke="#64748b" />
        <Tooltip
          contentStyle={{
            backgroundColor: '#1e293b',
            border: '1px solid #334155',
            borderRadius: '8px'
          }}
        />
        <Legend />

        {anomalyRegions.map((region, i) => (
          <ReferenceArea
            key={i}
            x1={region.start}
            x2={region.end}
            fill={selectedRegion?.start === region.start ? '#ef4444' : '#f87171'}
            fillOpacity={selectedRegion?.start === region.start ? 0.5 : 0.2}
            onClick={() => onRegionClick(region.start, region.end)}
            style={{ cursor: 'pointer' }}
          />
        ))}

        <Line
          type="monotone"
          dataKey="actual"
          stroke="#3b82f6"
          dot={false}
          name="Actual"
          strokeWidth={2}
        />
        <Line
          type="monotone"
          dataKey="predicted"
          stroke="#22c55e"
          strokeDasharray="5 5"
          dot={false}
          name="Predicted"
          strokeWidth={2}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
```

**Step 2:** Commit

```bash
git add frontend/src/components/SensorChart.tsx
git commit -m "feat: add SensorChart component with Recharts"
```

---

### Task 3.5: Create ChatWindow Component

**Files:**
- Create: `frontend/src/components/ChatWindow.tsx`

**Step 1:** Create component

```typescript
// frontend/src/components/ChatWindow.tsx
interface Props {
  diagnosis: string | null;
  loading: boolean;
  error: string | null;
  anomalyScore: number | null;
  sensorContributions: Record<string, number> | null;
}

export function ChatWindow({ diagnosis, loading, error, anomalyScore, sensorContributions }: Props) {
  return (
    <div className="chat-window">
      <h3 className="chat-title">Diagnostic Copilot</h3>

      {loading && (
        <div className="chat-bubble loading">
          <div className="loading-dots">
            <span></span><span></span><span></span>
          </div>
          Analyzing anomaly...
        </div>
      )}

      {error && (
        <div className="chat-bubble error">
          <strong>Error:</strong> {error}
        </div>
      )}

      {!loading && !error && !diagnosis && (
        <div className="chat-bubble hint">
          Click a red anomaly region on the chart to get a diagnosis
        </div>
      )}

      {anomalyScore !== null && sensorContributions && (
        <div className="chat-bubble stats">
          <div className="stat-row">
            <span>Anomaly Score:</span>
            <span className={anomalyScore > 0.7 ? 'high' : 'normal'}>
              {anomalyScore.toFixed(4)}
            </span>
          </div>
          <div className="stat-row">
            <span>Top Contributors:</span>
          </div>
          <ul className="contributors">
            {Object.entries(sensorContributions).map(([sensor, pct]) => (
              <li key={sensor}>
                {sensor}: {(pct * 100).toFixed(1)}%
              </li>
            ))}
          </ul>
        </div>
      )}

      {diagnosis && (
        <div className="chat-bubble diagnosis">
          <strong>Diagnosis:</strong>
          <p>{diagnosis}</p>
        </div>
      )}
    </div>
  );
}
```

**Step 2:** Commit

```bash
git add frontend/src/components/ChatWindow.tsx
git commit -m "feat: add ChatWindow component for diagnosis display"
```

---

### Task 3.6: Create Sidebar Component

**Files:**
- Create: `frontend/src/components/Sidebar.tsx`

**Step 1:** Create component

```typescript
// frontend/src/components/Sidebar.tsx
interface Props {
  threshold: number;
  onThresholdChange: (value: number) => void;
  engineId: string;
}

export function Sidebar({ threshold, onThresholdChange, engineId }: Props) {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h1>LEAP-Guard 360</h1>
        <p>Predictive Maintenance</p>
      </div>

      <div className="sidebar-section">
        <h3>Engine</h3>
        <select className="engine-select" disabled>
          <option>{engineId}</option>
        </select>
        <p className="hint-text">Multi-engine support coming soon</p>
      </div>

      <div className="sidebar-section">
        <h3>Anomaly Threshold</h3>
        <input
          type="range"
          min="0.5"
          max="1.0"
          step="0.05"
          value={threshold}
          onChange={(e) => onThresholdChange(parseFloat(e.target.value))}
          className="threshold-slider"
        />
        <div className="threshold-labels">
          <span>Sensitive</span>
          <span className="threshold-value">{threshold.toFixed(2)}</span>
          <span>Strict</span>
        </div>
      </div>

      <div className="sidebar-section">
        <h3>Legend</h3>
        <div className="legend-item">
          <span className="legend-line actual"></span>
          <span>Actual Reading</span>
        </div>
        <div className="legend-item">
          <span className="legend-line predicted"></span>
          <span>Predicted (Model)</span>
        </div>
        <div className="legend-item">
          <span className="legend-box anomaly"></span>
          <span>Anomaly Region</span>
        </div>
      </div>
    </aside>
  );
}
```

**Step 2:** Commit

```bash
git add frontend/src/components/Sidebar.tsx
git commit -m "feat: add Sidebar component with threshold control"
```

---

### Task 3.7: Create Test Data

**Files:**
- Create: `frontend/src/data/testData.ts`

**Step 1:** Create mock data generator

```typescript
// frontend/src/data/testData.ts
import { EngineData, AnomalyRegion } from '../types/api';

// Generate synthetic sensor data with degradation pattern
function generateSensorData(totalCycles: number): number[][] {
  const data: number[][] = [];

  for (let i = 0; i < totalCycles; i++) {
    const degradation = Math.pow(i / totalCycles, 2); // Exponential degradation
    const noise = () => (Math.random() - 0.5) * 0.1;

    // 13 sensor values (matching model features after variance filtering)
    const reading = [
      0.5 + degradation * 0.3 + noise(),  // s1
      0.5 + noise(),                       // s2 (stable)
      0.5 + degradation * 0.2 + noise(),  // s5
      0.5 + degradation * 0.15 + noise(), // s6
      0.5 + degradation * 0.4 + noise(),  // s7 (primary degradation signal)
      0.5 + degradation * 0.25 + noise(), // s9
      0.5 + degradation * 0.1 + noise(),  // s10
      0.5 + degradation * 0.2 + noise(),  // s11
      0.5 + degradation * 0.35 + noise(), // s12
      0.5 + noise(),                       // s13 (stable)
      0.5 + degradation * 0.15 + noise(), // s15
      0.5 + noise(),                       // s18 (stable)
      0.5 + noise(),                       // s19 (stable)
    ];

    data.push(reading);
  }

  return data;
}

// Generate chart data (actual vs predicted)
export function generateChartData(totalCycles: number) {
  const data = [];

  for (let i = 0; i < totalCycles; i++) {
    const degradation = Math.pow(i / totalCycles, 2);
    const noise = (Math.random() - 0.5) * 0.05;

    data.push({
      cycle: i + 1,
      actual: 0.5 + degradation * 0.4 + noise,
      predicted: 0.5 + degradation * 0.35, // Slightly behind actual (model lag)
    });
  }

  return data;
}

// Identify anomaly regions (where reconstruction error is high)
export function identifyAnomalyRegions(
  chartData: { cycle: number; actual: number; predicted: number }[],
  threshold: number = 0.05
): AnomalyRegion[] {
  const regions: AnomalyRegion[] = [];
  let regionStart: number | null = null;

  for (const point of chartData) {
    const error = Math.abs(point.actual - point.predicted);

    if (error > threshold && regionStart === null) {
      regionStart = point.cycle;
    } else if (error <= threshold && regionStart !== null) {
      regions.push({ start: regionStart, end: point.cycle - 1 });
      regionStart = null;
    }
  }

  // Close final region if still open
  if (regionStart !== null) {
    regions.push({ start: regionStart, end: chartData[chartData.length - 1].cycle });
  }

  return regions;
}

// Get sensor readings for a window
export function getWindowReadings(
  startCycle: number,
  windowSize: number = 50
): number[][] {
  const totalCycles = 250;
  const allData = generateSensorData(totalCycles);

  const startIdx = Math.max(0, startCycle - windowSize);
  const endIdx = Math.min(totalCycles, startCycle);

  return allData.slice(startIdx, endIdx);
}

export const testEngineData: EngineData = {
  engine_id: "LEAP-1A-001",
  metadata: {
    aircraft_type: "A320neo",
    total_cycles: 250
  },
  cycles: Array.from({ length: 250 }, (_, i) => ({
    cycle: i + 1,
    sensors: {
      T24: 642.15 + Math.random() * 10,
      T30: 1589.70 + Math.random() * 20,
      T50: 1406.36 + Math.random() * 15,
      P30: 554.36 + Math.random() * 5,
      Nf: 2388.06 + Math.random() * 10,
      Nc: 9046.19 + Math.random() * 20,
      Ps30: 47.47 + Math.random() * 2,
      phi: 521.66 + Math.random() * 5,
    }
  }))
};
```

**Step 2:** Commit

```bash
git add frontend/src/data/
git commit -m "feat: add test data generator for development"
```

---

### Task 3.8: Create Main App Component

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/App.css`

**Step 1:** Update App.tsx

```typescript
// frontend/src/App.tsx
import { useState, useCallback, useMemo } from 'react';
import { Sidebar } from './components/Sidebar';
import { SensorChart } from './components/SensorChart';
import { ChatWindow } from './components/ChatWindow';
import { useInference } from './hooks/useInference';
import { generateChartData, identifyAnomalyRegions, getWindowReadings } from './data/testData';
import { AnomalyRegion } from './types/api';
import './App.css';

const TOTAL_CYCLES = 250;
const WINDOW_SIZE = 50;

function App() {
  const [threshold, setThreshold] = useState(0.7);
  const [selectedRegion, setSelectedRegion] = useState<AnomalyRegion | null>(null);
  const { predict, loading, error, result, reset } = useInference();

  const chartData = useMemo(() => generateChartData(TOTAL_CYCLES), []);
  const anomalyRegions = useMemo(
    () => identifyAnomalyRegions(chartData, 0.05),
    [chartData]
  );

  const handleRegionClick = useCallback(async (start: number, end: number) => {
    setSelectedRegion({ start, end });
    reset();

    const readings = getWindowReadings(end, WINDOW_SIZE);

    if (readings.length < WINDOW_SIZE) {
      return; // Not enough data
    }

    await predict({
      sensor_readings: readings,
      window_size: WINDOW_SIZE,
      threshold
    });
  }, [predict, reset, threshold]);

  const handleThresholdChange = useCallback((value: number) => {
    setThreshold(value);
    setSelectedRegion(null);
    reset();
  }, [reset]);

  return (
    <div className="dashboard">
      <Sidebar
        threshold={threshold}
        onThresholdChange={handleThresholdChange}
        engineId="LEAP-1A-001"
      />

      <main className="main-content">
        <div className="chart-container">
          <h2>Engine Health Monitor</h2>
          <SensorChart
            data={chartData}
            anomalyRegions={anomalyRegions}
            onRegionClick={handleRegionClick}
            selectedRegion={selectedRegion}
          />
        </div>
      </main>

      <ChatWindow
        diagnosis={result?.diagnosis ?? null}
        loading={loading}
        error={error}
        anomalyScore={result?.anomaly_score ?? null}
        sensorContributions={result?.sensor_contributions ?? null}
      />
    </div>
  );
}

export default App;
```

**Step 2:** Update App.css (Apple-inspired design from approved mockups)

```css
/* frontend/src/App.css - Design tokens from docs/leap-guard-design.pen */

/* Light Theme (Landing Page) */
:root {
  --light-bg-primary: #FFFFFF;
  --light-bg-secondary: #F5F5F7;
  --light-bg-tertiary: #E5E5E7;
  --light-text-primary: #1D1D1F;
  --light-text-secondary: #86868B;

  /* Shared accent colors */
  --accent-blue: #007AFF;
  --success-green: #32D583;
  --warning-orange: #FF9900;
  --alert-red: #E85A4F;
}

/* Dark Theme (Dashboard) */
.dashboard {
  --bg-primary: #0D0D0D;
  --bg-secondary: #1A1A1A;
  --bg-tertiary: #2A2A2A;
  --text-primary: #FFFFFF;
  --text-secondary: #8E8E93;
  --border: #3A3A3A;
}

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  background: var(--bg-primary);
  color: var(--text-primary);
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  min-width: 1200px;
}

.dashboard {
  display: grid;
  grid-template-columns: 280px 1fr 320px;
  gap: 1rem;
  padding: 1rem;
  height: 100vh;
}

/* Sidebar */
.sidebar {
  background: var(--bg-secondary);
  border-radius: 12px;
  padding: 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.sidebar-header h1 {
  font-size: 1.5rem;
  color: var(--accent-blue);
}

.sidebar-header p {
  color: var(--text-secondary);
  font-size: 0.875rem;
}

.sidebar-section h3 {
  font-size: 0.875rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-secondary);
  margin-bottom: 0.75rem;
}

.engine-select {
  width: 100%;
  padding: 0.5rem;
  background: var(--bg-tertiary);
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--text-primary);
}

.hint-text {
  font-size: 0.75rem;
  color: var(--text-secondary);
  margin-top: 0.5rem;
}

.threshold-slider {
  width: 100%;
  accent-color: var(--accent-blue);
}

.threshold-labels {
  display: flex;
  justify-content: space-between;
  font-size: 0.75rem;
  color: var(--text-secondary);
  margin-top: 0.5rem;
}

.threshold-value {
  color: var(--accent-blue);
  font-weight: 600;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
  font-size: 0.875rem;
}

.legend-line {
  width: 24px;
  height: 3px;
  border-radius: 2px;
}

.legend-line.actual {
  background: var(--accent-blue);
}

.legend-line.predicted {
  background: var(--success-green);
  border-style: dashed;
}

.legend-box {
  width: 16px;
  height: 16px;
  border-radius: 4px;
}

.legend-box.anomaly {
  background: var(--alert-red);
  opacity: 0.5;
}

/* Main Content */
.main-content {
  display: flex;
  flex-direction: column;
}

.chart-container {
  background: var(--bg-secondary);
  border-radius: 12px;
  padding: 1.5rem;
  flex: 1;
}

.chart-container h2 {
  margin-bottom: 1rem;
  font-size: 1.25rem;
}

/* Chat Window */
.chat-window {
  background: var(--bg-secondary);
  border-radius: 12px;
  padding: 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.chat-title {
  font-size: 1rem;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.chat-bubble {
  background: var(--bg-tertiary);
  border-radius: 12px;
  padding: 1rem;
}

.chat-bubble.hint {
  color: var(--text-secondary);
  font-style: italic;
}

.chat-bubble.error {
  border-left: 3px solid var(--alert-red);
}

.chat-bubble.diagnosis {
  border-left: 3px solid var(--accent-blue);
}

.chat-bubble.diagnosis p {
  margin-top: 0.5rem;
  line-height: 1.6;
}

.chat-bubble.stats {
  font-size: 0.875rem;
}

.stat-row {
  display: flex;
  justify-content: space-between;
  margin-bottom: 0.5rem;
}

.stat-row .high {
  color: var(--alert-red);
  font-weight: 600;
}

.stat-row .normal {
  color: var(--success-green);
}

.contributors {
  list-style: none;
  margin-top: 0.5rem;
  padding-left: 1rem;
}

.contributors li {
  color: var(--text-secondary);
  margin-bottom: 0.25rem;
}

/* Loading animation */
.loading-dots {
  display: inline-flex;
  gap: 4px;
  margin-right: 8px;
}

.loading-dots span {
  width: 6px;
  height: 6px;
  background: var(--accent-blue);
  border-radius: 50%;
  animation: bounce 1.4s infinite ease-in-out both;
}

.loading-dots span:nth-child(1) { animation-delay: -0.32s; }
.loading-dots span:nth-child(2) { animation-delay: -0.16s; }

@keyframes bounce {
  0%, 80%, 100% { transform: scale(0); }
  40% { transform: scale(1); }
}
```

**Step 3:** Verify app runs

Run: `cd frontend && npm run dev`
Expected: Dashboard renders with chart, sidebar, and chat window

**Step 4:** Commit

```bash
git add frontend/src/App.tsx frontend/src/App.css
git commit -m "feat: implement main dashboard layout with dark theme"
```

---

### Task 3.9: Create Production Environment

**Files:**
- Create: `frontend/.env.production`

**Step 1:** Create production env file

```
# frontend/.env.production
VITE_API_URL=https://YOUR_LAMBDA_FUNCTION_URL
```

**Step 2:** Update with actual URL after deploying backend

Run: Get URL from `sam deploy` output or:
```bash
aws cloudformation describe-stacks \
  --stack-name leap-guard-inference \
  --query "Stacks[0].Outputs[?OutputKey=='FunctionUrl'].OutputValue" \
  --output text
```

**Step 3:** Build for production

Run: `cd frontend && npm run build`
Expected: Creates `dist/` folder with optimized assets

**Step 4:** Commit

```bash
git add frontend/.env.production
git commit -m "chore: add production environment config"
```

---

### Task 3.10: Deploy Frontend to S3

**Step 1:** Create S3 bucket

```bash
aws s3 mb s3://leap-guard-frontend-$(aws sts get-caller-identity --query Account --output text) --region ap-southeast-1
```

**Step 2:** Enable static website hosting

```bash
BUCKET=leap-guard-frontend-$(aws sts get-caller-identity --query Account --output text)
aws s3 website s3://$BUCKET --index-document index.html --error-document index.html
```

**Step 3:** Upload build

```bash
cd frontend
npm run build
aws s3 sync dist/ s3://$BUCKET --delete
```

**Step 4:** Set public read policy

```bash
aws s3api put-bucket-policy --bucket $BUCKET --policy '{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": "*",
    "Action": "s3:GetObject",
    "Resource": "arn:aws:s3:::'$BUCKET'/*"
  }]
}'
```

**Step 5:** Get website URL

```bash
echo "http://$BUCKET.s3-website-ap-southeast-1.amazonaws.com"
```

**Step 6:** Commit deployment scripts

```bash
git add .
git commit -m "chore: complete frontend deployment"
```

---

## Summary

| Phase | Tasks | Est. Time |
|-------|-------|-----------|
| Phase 1 (Colab) | 1.1-1.6 | 2-3 hours |
| Phase 2 (Backend) | 2.1-2.8 | 3-4 hours |
| Phase 3 (Frontend) | 3.1-3.10 | 2-3 hours |
| **Total** | **24 tasks** | **7-10 hours** |

---

## Verification Checklists

### Phase 1
- [ ] Model trains without errors
- [ ] Threshold computed (95th percentile)
- [ ] All 4 artifacts exported

### Phase 2
- [ ] All tests pass locally
- [ ] Docker container builds
- [ ] Lambda deployed with Function URL
- [ ] Bedrock returns diagnoses

### Phase 3
- [ ] Dev server runs (`npm run dev`)
- [ ] Landing page matches approved design (light theme, Apple-style)
- [ ] Dashboard matches approved design (dark theme, 3-column layout)
- [ ] Chart renders with anomaly regions highlighted
- [ ] Click anomaly triggers API call and shows diagnosis
- [ ] AI Diagnostic Copilot panel shows sensor contributions
- [ ] Deployed to S3 with CORS working
