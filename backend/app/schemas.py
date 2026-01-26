from pydantic import BaseModel, Field, field_validator
from typing import Optional


class PredictRequest(BaseModel):
    sensor_readings: list[list[float]]
    window_size: int = Field(default=10, ge=10, le=100)
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
