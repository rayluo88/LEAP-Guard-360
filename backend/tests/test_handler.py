# backend/tests/test_handler.py
import pytest
import json
import os

os.environ["MOCK_BEDROCK"] = "true"

from app.handler import lambda_handler

# Match actual model config: 8 features, window_size 10
N_FEATURES = 8
WINDOW_SIZE = 10


def test_handler_valid_request():
    event = {
        "body": json.dumps({
            "sensor_readings": [[0.5] * N_FEATURES] * WINDOW_SIZE,
            "window_size": WINDOW_SIZE,
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


def test_handler_anomaly_triggers_diagnosis():
    # Create data that should trigger high reconstruction error (anomaly)
    # Use extreme values that differ from training distribution
    extreme_readings = [[0.1 if i % 2 == 0 else 0.9 for i in range(N_FEATURES)]] * WINDOW_SIZE
    event = {
        "body": json.dumps({
            "sensor_readings": extreme_readings,
            "window_size": WINDOW_SIZE,
            "threshold": 0.001  # Very low threshold to ensure anomaly is triggered
        })
    }
    response = lambda_handler(event, None)

    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert body["is_anomaly"] is True
    assert body["diagnosis"] is not None  # Mock diagnosis should be returned
