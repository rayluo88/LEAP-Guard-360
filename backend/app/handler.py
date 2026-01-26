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
