# backend/tests/test_inference.py
import pytest
import numpy as np
from app.inference import AnomalyDetector

# Actual model config: 8 features, window_size 10
N_FEATURES = 8
WINDOW_SIZE = 10

@pytest.fixture
def detector():
    return AnomalyDetector(model_dir="model/")

def test_detector_loads(detector):
    assert detector.model is not None
    assert detector.scaler is not None
    assert detector.default_threshold > 0

def test_predict_healthy_data(detector):
    # Generate healthy-looking data (normalized, low variance)
    readings = np.random.uniform(0.3, 0.7, (WINDOW_SIZE, N_FEATURES)).tolist()
    result = detector.predict(readings, window_size=WINDOW_SIZE)

    assert "anomaly_score" in result
    assert "is_anomaly" in result
    assert "sensor_contributions" in result
    assert isinstance(result["anomaly_score"], float)

def test_predict_returns_top_3_contributors(detector):
    readings = np.random.uniform(0.3, 0.7, (WINDOW_SIZE, N_FEATURES)).tolist()
    result = detector.predict(readings, window_size=WINDOW_SIZE)

    assert len(result["sensor_contributions"]) == 3

def test_predict_invalid_shape(detector):
    readings = [[1.0] * 5] * 5  # Wrong shape (5x5 instead of 10x8)
    with pytest.raises(ValueError, match="Expected shape"):
        detector.predict(readings, window_size=WINDOW_SIZE)

def test_predict_with_custom_threshold(detector):
    readings = np.random.uniform(0.3, 0.7, (WINDOW_SIZE, N_FEATURES)).tolist()
    result = detector.predict(readings, window_size=WINDOW_SIZE, threshold=0.5)

    # Result should use the custom threshold for comparison
    assert isinstance(result["is_anomaly"], bool)
