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
