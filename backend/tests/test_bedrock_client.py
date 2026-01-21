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
        sensor_values=[0.5] * 8,
        anomaly_score=0.85,
        top_contributors={"T3_HPC_Temp": 0.4, "P3_HPC_Pressure": 0.3, "N2_Core_RPM": 0.2}
    )
    assert isinstance(result, str)
    assert "T3_HPC_Temp" in result  # Top contributor mentioned

def test_sensor_descriptions_exist(mock_client):
    # Actual feature names from model config
    assert "T3_HPC_Temp" in mock_client.sensor_descriptions
    assert "P3_HPC_Pressure" in mock_client.sensor_descriptions
    assert "N1_Fan_RPM" in mock_client.sensor_descriptions
