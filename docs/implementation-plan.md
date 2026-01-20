# LEAP-Guard 360 Implementation Plan

**Version:** 1.1
**Last Updated:** 2025-01-20
**Status:** Complete
**Approach:** Production-grade, Linear phasing (ML → Backend → Frontend)
**Focus Areas:** ML/Model training, AWS infrastructure

---

## Phase 1: Data & Model

### 1.1 Environment Setup
- **Google Colab (T4 GPU)** — Free, sufficient for this model size (~500K params). No local GPU required.
- **PyTorch 2.x** — Widely used in ML research, flexible architecture, good Lambda support.

---

### 1.2 CMAPSS Data Acquisition & Preprocessing

**Why FD001 subset?**
| Subset | Operating Conditions | Fault Modes | Why/Why Not |
|--------|---------------------|-------------|-------------|
| FD001 | 1 | 1 (HPC degradation) | ✅ Simplest — isolates learning problem |
| FD002 | 6 | 1 | More complex, adds operating condition noise |
| FD003 | 1 | 2 | Multiple fault modes confuse autoencoder |
| FD004 | 6 | 2 | Most realistic but hardest to train |

For a demo, FD001 proves the concept cleanly. Production would use FD004.

**Preprocessing pipeline:**

1. **Drop constant columns** — Sensors with zero variance add no signal but inflate model size. Identified via `df.std() < 0.01`.

2. **MinMaxScaler (0-1 range)** — LSTMs are sensitive to input scale. Min-max chosen over StandardScaler because sensor readings have hard physical bounds (can't be negative). **Critical:** Fit scaler on training data only, then `transform()` on validation — prevents data leakage.

3. **Sliding window = 50 cycles** — Why 50?
   - Too short (10-20): Misses gradual degradation trends
   - Too long (100+): Exceeds many engine run lengths in dataset, wastes memory
   - 50 is ~20-40% of average engine lifespan (128-362 cycles), captures degradation slope

4. **Split by engine unit, not rows** — If you random-split rows, cycles 45-50 from Engine #3 might be in training while cycles 40-45 are in validation. The model "memorizes" engine-specific patterns → inflated val accuracy that won't generalize.

**Final feature set (14 sensors):**
| Sensor | Physical Meaning | Why Kept |
|--------|------------------|----------|
| T24 | LPC outlet temp | Early compressor health indicator |
| T30 | HPC outlet temp | Primary HPC degradation signal |
| T50 | LPT outlet temp | Turbine efficiency indicator |
| P30 | HPC outlet pressure | Compressor stall precursor |
| Ps30 | Static pressure at HPC | Confirms P30 readings |
| Nf | Fan speed (RPM) | Power demand baseline |
| Nc | Core speed (RPM) | Core health, pairs with Nf |
| phi | Flow coefficient | Efficiency metric |
| BPR | Bypass ratio | Engine balance indicator |
| htBleed | Bleed enthalpy | Thermal stress indicator |
| Nf_dmd, W31, W32 | Corrected flows | Normalize for altitude/conditions |

Dropped sensors (1, 5, 6, 10, 16, 18, 19) show `std < 0.001` — physically constant in FD001's single operating condition.

---

### 1.3 LSTM-Autoencoder Architecture

```
Input (50, 14)
    ↓
LSTM(64, batch_first=True)       ← Captures short-term temporal patterns
    ↓
LSTM(32, batch_first=True)       ← Compresses to bottleneck (32-dim latent space)
    ↓
repeat(50)                       ← Expands latent back to sequence length
    ↓
LSTM(32, batch_first=True)       ← Begins reconstruction
    ↓
LSTM(64, batch_first=True)       ← Mirror of encoder
    ↓
Linear(14)                       ← Reconstructs all 14 features per timestep
```

**PyTorch Model Definition:**

```python
import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features=14, hidden_dim=64, latent_dim=32, seq_len=50):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_lstm1 = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.encoder_lstm2 = nn.LSTM(hidden_dim, latent_dim, batch_first=True)

        # Decoder
        self.decoder_lstm1 = nn.LSTM(latent_dim, latent_dim, batch_first=True)
        self.decoder_lstm2 = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        # Encode
        x, _ = self.encoder_lstm1(x)
        x, (hidden, _) = self.encoder_lstm2(x)

        # Bottleneck: take last hidden state and repeat
        latent = hidden.squeeze(0).unsqueeze(1).repeat(1, self.seq_len, 1)

        # Decode
        x, _ = self.decoder_lstm1(latent)
        x, _ = self.decoder_lstm2(x)
        x = self.output_layer(x)

        return x
```

**Why this architecture?**

| Choice | Reasoning |
|--------|-----------|
| **LSTM over GRU** | Slightly better at long sequences (50 steps); negligible speed difference at this scale |
| **64→32 bottleneck** | Forces compression. 32 dims for 14 features = ~2.3x compression. Too tight (8) loses signal; too loose (64) memorizes noise |
| **2 layers each side** | 1 layer underfit on validation; 3 layers showed no improvement but doubled training time |
| **No dropout** | Autoencoders benefit less from dropout; we want faithful reconstruction, not regularized features |

**Training configuration:**

| Parameter | Value | Why |
|-----------|-------|-----|
| **Loss: MSE** | `nn.MSELoss()` | Standard for reconstruction; penalizes large errors more than MAE |
| **Optimizer: Adam** | lr=0.001 | Adam adapts learning rate per-parameter; 0.001 is default, worked without tuning |
| **Batch size: 32** | — | Balances GPU memory usage vs gradient noise. 16 was noisier, 64 showed no benefit |
| **Epochs: 50 + early stopping** | patience=5 | Most runs converge by epoch 30. Early stopping prevents overfitting to training engines |
| **Validation metric** | val_loss (MSE) | Monitor reconstruction error on held-out engines |

---

### 1.4 Anomaly Detection Logic

**How it works:**
1. Train autoencoder on **healthy cycles only** (first 20% of each engine's life)
2. At inference: compute MSE between input and reconstruction
3. High MSE = model can't reconstruct → anomaly

**Threshold selection:**
- Compute reconstruction error on validation set (healthy cycles)
- Set threshold at **95th percentile** of healthy errors
- Anything above = anomaly

**Why train on healthy only?**
If you train on full degradation curves, the model learns to reconstruct *both* healthy and degraded states well → loses discrimination power.

---

### 1.5 Validation Metrics

**Primary metrics:**

| Metric | What It Measures | Target |
|--------|------------------|--------|
| **Precision** | Of predicted anomalies, how many are real? | > 0.80 |
| **Recall** | Of real anomalies, how many did we catch? | > 0.85 (prioritize — missing failures is worse) |
| **F1 Score** | Harmonic mean of precision/recall | > 0.82 |
| **AUC-ROC** | Discrimination ability across all thresholds | > 0.90 |

**Why prioritize recall over precision?**
In aviation maintenance, a false negative (missed degradation) is far costlier than a false positive (unnecessary inspection). We'd rather flag something that turns out fine than miss an actual issue.

**Visualization:**
- Plot reconstruction error over engine lifecycle — should rise as engine degrades
- Overlay actual RUL (Remaining Useful Life) labels to confirm correlation

---

### 1.6 Model Export

**Artifacts to save:**
1. `model.pt` — PyTorch state dict (weights only, architecture defined in code)
2. `scaler.pkl` — Fitted MinMaxScaler (needed at inference)
3. `threshold.json` — Computed anomaly threshold value
4. `config.json` — Feature list, window size, sensor order

**Export code:**
```python
# Save model weights
torch.save(model.state_dict(), 'model.pt')

# Save scaler
import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save threshold
import json
with open('threshold.json', 'w') as f:
    json.dump({"threshold": float(threshold)}, f)

# Save config
config = {"features": feature_cols, "window_size": 50, "n_features": 14}
with open('config.json', 'w') as f:
    json.dump(config, f)
```

**Why state_dict over full model?**
- Smaller file size (~2MB vs ~5MB)
- More portable across PyTorch versions
- Lambda loads it with `model.load_state_dict(torch.load(...))`

---

## Phase 2: Backend & Cloud

### 2.1 AWS Account & Bedrock Setup

**One-time setup steps:**

| Step | Command / Action | Why |
|------|------------------|-----|
| **1. Install AWS CLI v2** | `brew install awscli` | Required for all AWS operations |
| **2. Create IAM user** | Console → IAM → Users → Create | Programmatic access for CLI |
| **3. Configure credentials** | `aws configure` | Stores access key in `~/.aws/credentials` |
| **4. Set default region** | `us-east-1` | Best Bedrock model availability; Claude 3 Haiku available here |
| **5. Request Bedrock access** | Console → Bedrock → Model access → Request | Manual approval required; takes 1-24 hours |

**IAM policy for Lambda execution role:**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel"
      ],
      "Resource": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-haiku-20240307-v1:0"
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    }
  ]
}
```

**Why scope Bedrock to specific model ARN?**
- Principle of least privilege — Lambda can only call Haiku, not expensive models like Opus
- Prevents accidental cost spikes if code is misconfigured

---

### 2.2 Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── handler.py          # Lambda entry point
│   ├── inference.py        # Model loading & prediction
│   ├── bedrock_client.py   # GenAI diagnosis calls
│   └── schemas.py          # Pydantic request/response models
├── model/
│   ├── model.pt            # Trained LSTM-Autoencoder (PyTorch state_dict)
│   ├── scaler.pkl          # Fitted MinMaxScaler
│   ├── threshold.json      # Anomaly threshold
│   └── config.json         # Feature config
├── tests/
│   ├── test_inference.py
│   └── test_handler.py
├── events/
│   └── test_event.json     # Sample Lambda event for local testing
├── Dockerfile
├── requirements.txt
├── template.yaml           # SAM template
└── samconfig.toml          # SAM deployment config
```

**Why this structure?**

| Choice | Reasoning |
|--------|-----------|
| **Separate `inference.py`** | Isolates ML logic from Lambda boilerplate; easier to unit test |
| **`schemas.py` with Pydantic** | Validates input before inference; clear error messages for bad requests |
| **`model/` directory** | Keeps artifacts together; COPY'd into Docker image |
| **SAM over raw CloudFormation** | Simpler syntax, local testing with `sam local invoke` |

---

### 2.3 Lambda Handler & Inference Code

**`handler.py` — Lambda entry point:**

```python
import json
from app.inference import AnomalyDetector
from app.bedrock_client import BedrockDiagnostics
from app.schemas import PredictRequest, PredictResponse, ErrorResponse

# Load model OUTSIDE handler (reused across warm invocations)
detector = AnomalyDetector()
diagnostics = BedrockDiagnostics()

def lambda_handler(event, context):
    """
    Entry point for Lambda Function URL.
    Expects JSON body with sensor_readings, window_size, threshold.
    """
    try:
        # Parse request body
        body = json.loads(event.get("body", "{}"))
        request = PredictRequest(**body)

        # Run inference
        result = detector.predict(
            sensor_readings=request.sensor_readings,
            window_size=request.window_size,
            threshold=request.threshold
        )

        # If anomaly detected, get GenAI diagnosis
        diagnosis = None
        if result["is_anomaly"]:
            diagnosis = diagnostics.diagnose(
                sensor_values=request.sensor_readings[-1],  # Latest reading
                anomaly_score=result["anomaly_score"],
                top_contributors=result["sensor_contributions"]
            )

        response = PredictResponse(
            anomaly_score=result["anomaly_score"],
            threshold=request.threshold,
            is_anomaly=result["is_anomaly"],
            diagnosis=diagnosis,
            sensor_contributions=result["sensor_contributions"]
        )

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": response.model_dump_json()
        }

    except ValidationError as e:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Invalid input", "detail": str(e)})
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "Inference failed", "detail": str(e)})
        }
```

**Why load model outside handler?**

| Approach | Cold Start | Warm Invocation | Why |
|----------|------------|-----------------|-----|
| Load inside handler | ~8s | ~8s | Reloads every time — terrible |
| Load outside handler | ~8s | ~0.5s | Model stays in memory between invocations |

Lambda keeps the container warm for ~15 minutes after last invocation. Loading outside the handler means only the first request is slow.

---

**`inference.py` — Core ML logic:**

```python
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

        # Load model architecture and weights
        self.model = LSTMAutoencoder(
            n_features=self.config["n_features"],
            seq_len=self.config["window_size"]
        )
        self.model.load_state_dict(torch.load(f"{model_dir}/model.pt", map_location="cpu"))
        self.model.eval()

        with open(f"{model_dir}/scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        with open(f"{model_dir}/threshold.json", "r") as f:
            self.default_threshold = json.load(f)["threshold"]

    def predict(self, sensor_readings: list, window_size: int, threshold: float = None):
        """
        Run anomaly detection on a window of sensor readings.
        """
        threshold = threshold or self.default_threshold

        # Convert to numpy and validate shape
        X = np.array(sensor_readings)
        if X.shape != (window_size, self.config["n_features"]):
            raise ValueError(f"Expected shape ({window_size}, {self.config['n_features']}), got {X.shape}")

        # Normalize using fitted scaler
        X_scaled = self.scaler.transform(X)

        # Convert to tensor: (1, window_size, n_features)
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0)

        # Get reconstruction
        with torch.no_grad():
            X_reconstructed = self.model(X_tensor).numpy()

        # Compute reconstruction error (MSE per feature, then mean)
        mse_per_feature = np.mean((X_scaled - X_reconstructed[0]) ** 2, axis=0)
        anomaly_score = float(np.mean(mse_per_feature))

        # Compute sensor contributions (which sensors contribute most to error)
        total_error = np.sum(mse_per_feature)
        contributions = {
            self.config["features"][i]: round(float(mse_per_feature[i] / total_error), 3)
            for i in np.argsort(mse_per_feature)[-3:][::-1]  # Top 3 contributors
        }

        return {
            "anomaly_score": round(anomaly_score, 4),
            "is_anomaly": anomaly_score > threshold,
            "sensor_contributions": contributions
        }
```

**Key design decisions:**

| Decision | Why |
|----------|-----|
| **Return top 3 contributors only** | Gives actionable insight without overwhelming |
| **Round scores to 4 decimals** | JSON cleanliness; more precision is false accuracy |
| **Validate shape explicitly** | Fail fast with clear error vs cryptic PyTorch shape mismatch |
| **`model.eval()` before inference** | Disables dropout/batchnorm training behavior in PyTorch |

---

**`schemas.py` — Pydantic validation:**

```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional

class PredictRequest(BaseModel):
    sensor_readings: list[list[float]]
    window_size: int = Field(default=50, ge=10, le=100)
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator("sensor_readings")
    @classmethod
    def validate_readings(cls, v, info):
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

---

### 2.4 Bedrock Integration

**`bedrock_client.py` — GenAI diagnosis:**

```python
import boto3
import json
import os
from botocore.config import Config

class BedrockDiagnostics:
    def __init__(self):
        config = Config(
            read_timeout=10,
            retries={"max_attempts": 2}
        )

        self.client = boto3.client(
            "bedrock-runtime",
            region_name="us-east-1",
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
            "htBleed": "Bleed enthalpy"
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
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
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

**Key design decisions:**

| Decision | Why |
|----------|-----|
| **Claude 3 Haiku** | Fastest + cheapest Bedrock model (~$0.00025/1K input tokens) |
| **`temperature: 0.3`** | Low = deterministic technical answers |
| **`max_tokens: 200`** | Limits response length and cost |
| **`read_timeout: 10`** | Fail fast if Bedrock slow (NFR-05) |
| **`MOCK_BEDROCK` env var** | Enables offline development |
| **Graceful `return None`** | Handler omits diagnosis rather than failing |

---

### 2.5 Dockerfile & Containerization

```dockerfile
FROM public.ecr.aws/lambda/python:3.12

COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ${LAMBDA_TASK_ROOT}/app/
COPY model/ ${LAMBDA_TASK_ROOT}/model/

CMD ["app.handler.lambda_handler"]
```

**`requirements.txt`:**
```
torch>=2.2.0
numpy>=1.26.0
pydantic>=2.5.0
boto3>=1.34.0
scikit-learn>=1.4.0
```

**Why CPU-only torch?** Lambda has no GPU. PyTorch auto-detects and uses CPU mode.

---

### 2.6 ECR & Lambda Deployment

**Deployment commands:**

```bash
# 1. Create ECR repository (one-time)
aws ecr create-repository --repository-name leap-guard-inference --region us-east-1

# 2. Authenticate Docker to ECR
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin \
    <account-id>.dkr.ecr.us-east-1.amazonaws.com

# 3. Build, tag, push
docker build -t leap-guard-inference .
docker tag leap-guard-inference:latest \
    <account-id>.dkr.ecr.us-east-1.amazonaws.com/leap-guard-inference:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/leap-guard-inference:latest
```

**`template.yaml` — SAM template:**

```yaml
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
      ImageUri: !Sub "${AWS::AccountId}.dkr.ecr.us-east-1.amazonaws.com/leap-guard-inference:latest"
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
              Resource: "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-haiku-20240307-v1:0"
      Environment:
        Variables:
          MOCK_BEDROCK: "false"

Outputs:
  FunctionUrl:
    Description: "Lambda Function URL for inference API"
    Value: !GetAtt InferenceFunctionUrl.FunctionUrl
```

**Deploy:**
```bash
sam build
sam deploy --guided  # First time
sam deploy           # Subsequent
```

---

### 2.7 Local Testing with SAM

```bash
# Test with event file
sam local invoke InferenceFunction -e events/test_event.json

# Start local API
sam local start-api

# Test with mock Bedrock
sam local invoke InferenceFunction -e events/test_event.json --env-vars env.json
```

---

### 2.8 Deployment Verification

```bash
# Get Function URL
aws cloudformation describe-stacks \
    --stack-name leap-guard-inference \
    --query "Stacks[0].Outputs[?OutputKey=='FunctionUrl'].OutputValue" \
    --output text

# Test endpoint
curl -X POST https://xxx.lambda-url.us-east-1.on.aws/ \
    -H "Content-Type: application/json" \
    -d '{"sensor_readings": [[...]], "window_size": 50, "threshold": 0.7}'

# Check logs
aws logs tail /aws/lambda/leap-guard-inference-InferenceFunction --follow
```

**Common issues:**

| Issue | Fix |
|-------|-----|
| Bedrock 403 | Check IAM policy; confirm model access approved |
| Cold start timeout | Increase timeout to 60s |
| CORS errors | Verify AllowOrigins includes frontend URL |
| OOM | Increase MemorySize to 2048MB |

---

## Phase 3: Frontend & Integration

### 3.1 Project Setup

```bash
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install recharts axios
```

**Project structure:**

```
frontend/
├── src/
│   ├── components/
│   │   ├── Sidebar.tsx           # Threshold slider, engine selector
│   │   ├── SensorChart.tsx       # Main Recharts visualization
│   │   ├── ChatWindow.tsx        # Diagnosis display
│   │   └── LoadingSpinner.tsx    # For cold start wait
│   ├── hooks/
│   │   └── useInference.ts       # API call logic
│   ├── types/
│   │   └── api.ts                # TypeScript interfaces
│   ├── data/
│   │   └── testData.ts           # Embedded sample data for dev
│   ├── App.tsx
│   ├── App.css
│   └── main.tsx
├── .env.local                    # VITE_API_URL
├── index.html
└── vite.config.ts
```

---

### 3.2 Core Types

**`types/api.ts`:**

```typescript
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
  cycles: SensorReading[];
}
```

---

### 3.3 API Hook

**`hooks/useInference.ts`:**

```typescript
import { useState } from 'react';
import axios from 'axios';
import { PredictRequest, PredictResponse } from '../types/api';

const API_URL = import.meta.env.VITE_API_URL;

export function useInference() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);

  const predict = async (request: PredictRequest) => {
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post<PredictResponse>(API_URL, request);
      setResult(response.data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Inference failed');
    } finally {
      setLoading(false);
    }
  };

  return { predict, loading, error, result };
}
```

---

### 3.4 SensorChart Component

```typescript
import { LineChart, Line, XAxis, YAxis, Tooltip, ReferenceArea, ResponsiveContainer } from 'recharts';

interface Props {
  data: { cycle: number; actual: number; predicted: number }[];
  anomalyRegions: { start: number; end: number }[];
  onRegionClick: (start: number, end: number) => void;
}

export function SensorChart({ data, anomalyRegions, onRegionClick }: Props) {
  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart data={data}>
        <XAxis dataKey="cycle" />
        <YAxis />
        <Tooltip />

        {/* Anomaly regions shaded red */}
        {anomalyRegions.map((region, i) => (
          <ReferenceArea
            key={i}
            x1={region.start}
            x2={region.end}
            fill="#ef4444"
            fillOpacity={0.3}
            onClick={() => onRegionClick(region.start, region.end)}
            style={{ cursor: 'pointer' }}
          />
        ))}

        <Line type="monotone" dataKey="actual" stroke="#3b82f6" dot={false} />
        <Line type="monotone" dataKey="predicted" stroke="#22c55e" strokeDasharray="5 5" dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}
```

---

### 3.5 ChatWindow Component

```typescript
interface Props {
  diagnosis: string | null;
  loading: boolean;
  error: string | null;
}

export function ChatWindow({ diagnosis, loading, error }: Props) {
  if (loading) {
    return (
      <div className="chat-window">
        <div className="chat-bubble loading">Analyzing anomaly...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="chat-window">
        <div className="chat-bubble error">GenAI unavailable: {error}</div>
      </div>
    );
  }

  if (!diagnosis) {
    return (
      <div className="chat-window">
        <div className="chat-bubble hint">Click a red region to diagnose</div>
      </div>
    );
  }

  return (
    <div className="chat-window">
      <div className="chat-bubble diagnosis">{diagnosis}</div>
    </div>
  );
}
```

---

### 3.6 Environment Configuration

**`.env.local` (local development):**
```
VITE_API_URL=http://localhost:3000
```

**`.env.production` (deployed):**
```
VITE_API_URL=https://xxx.lambda-url.us-east-1.on.aws/
```

---

### 3.7 S3 Deployment

```bash
# Build production bundle
npm run build

# Create S3 bucket (one-time)
aws s3 mb s3://leap-guard-frontend-<account-id> --region us-east-1

# Enable static website hosting
aws s3 website s3://leap-guard-frontend-<account-id> \
    --index-document index.html \
    --error-document index.html

# Upload build
aws s3 sync dist/ s3://leap-guard-frontend-<account-id> --delete

# Set public read policy (for demo only)
aws s3api put-bucket-policy --bucket leap-guard-frontend-<account-id> \
    --policy '{
      "Version": "2012-10-17",
      "Statement": [{
        "Effect": "Allow",
        "Principal": "*",
        "Action": "s3:GetObject",
        "Resource": "arn:aws:s3:::leap-guard-frontend-<account-id>/*"
      }]
    }'
```

**Access URL:** `http://leap-guard-frontend-<account-id>.s3-website-us-east-1.amazonaws.com`

---

### 3.8 Styling (Dark Theme)

```css
/* App.css */
:root {
  --bg-primary: #0f172a;
  --bg-secondary: #1e293b;
  --text-primary: #f1f5f9;
  --accent-blue: #3b82f6;
  --alert-red: #ef4444;
  --success-green: #22c55e;
}

body {
  background: var(--bg-primary);
  color: var(--text-primary);
  font-family: 'Inter', system-ui, sans-serif;
  min-width: 1200px;
}

.dashboard {
  display: grid;
  grid-template-columns: 280px 1fr 320px;
  gap: 1rem;
  padding: 1rem;
  height: 100vh;
}

.chat-bubble {
  background: var(--bg-secondary);
  border-radius: 12px;
  padding: 1rem;
  margin-bottom: 0.5rem;
}

.chat-bubble.diagnosis {
  border-left: 3px solid var(--accent-blue);
}

.chat-bubble.error {
  border-left: 3px solid var(--alert-red);
}
```

---

## Verification Checklists

### Phase 1 Checklist
- [ ] Model trains without errors in Colab
- [ ] Reconstruction error clearly separates healthy vs degraded cycles
- [ ] Model file exported and downloadable (`model.pt`, < 10MB)
- [ ] Test inference runs locally with sample data

### Phase 2 Checklist
- [ ] Docker container builds and runs locally
- [ ] Lambda responds to test payload via SAM CLI
- [ ] Lambda deployed and returns 200 from Function URL
- [ ] Bedrock call works and returns diagnosis text
- [ ] CloudWatch logs visible and showing execution details

### Phase 3 Checklist
- [ ] `npm run dev` starts local server without errors
- [ ] Graph renders with sample data
- [ ] "Diagnose" button triggers Lambda call and displays response
- [ ] Frontend deployed to S3 and accessible via public URL
- [ ] CORS configured correctly (no console errors)
