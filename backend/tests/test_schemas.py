import pytest
from app.schemas import PredictRequest, PredictResponse

N_FEATURES = 8
WINDOW_SIZE = 10

def test_predict_request_valid():
    request = PredictRequest(
        sensor_readings=[[1.0] * N_FEATURES] * WINDOW_SIZE,
        window_size=WINDOW_SIZE,
        threshold=0.12
    )
    assert request.window_size == WINDOW_SIZE
    assert request.threshold == 0.12

def test_predict_request_defaults():
    request = PredictRequest(
        sensor_readings=[[1.0] * N_FEATURES] * WINDOW_SIZE
    )
    assert request.window_size == WINDOW_SIZE
    assert request.threshold is None

def test_predict_request_invalid_window():
    with pytest.raises(ValueError):
        PredictRequest(
            sensor_readings=[[1.0] * N_FEATURES] * WINDOW_SIZE,
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
