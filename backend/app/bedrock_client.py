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

        # Default to an inference-profile model that's already enabled in the account.
        # Override with BEDROCK_MODEL_ID if needed (e.g., Anthropic Claude).
        self.model_id = os.environ.get(
            "BEDROCK_MODEL_ID",
            "apac.amazon.nova-micro-v1:0"
        )
        self.mock_mode = os.environ.get("MOCK_BEDROCK", "false").lower() == "true"

        # Sensor descriptions matching actual model features
        self.sensor_descriptions = {
            "T25_LPC_Temp": "Low Pressure Compressor outlet temperature",
            "T3_HPC_Temp": "High Pressure Compressor outlet temperature",
            "T49_EGT": "Exhaust Gas Temperature",
            "P3_HPC_Pressure": "High Pressure Compressor outlet pressure",
            "N1_Fan_RPM": "Fan rotational speed (N1)",
            "N2_Core_RPM": "Core rotational speed (N2)",
            "N1_Vib_Units": "Fan vibration level",
            "Oil_Pressure_PSI": "Oil system pressure",
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
            if "anthropic" in self.model_id:
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

            response = self.client.converse(
                modelId=self.model_id,
                messages=[{
                    "role": "user",
                    "content": [{"text": prompt}]
                }],
                inferenceConfig={
                    "maxTokens": 200,
                    "temperature": 0.3
                }
            )
            return response["output"]["message"]["content"][0]["text"]
        except Exception as e:
            print(f"Bedrock error: {e}")
            return None

    def _mock_response(self, top_contributors: dict) -> str:
        top_sensor = list(top_contributors.keys())[0]
        desc = self.sensor_descriptions.get(top_sensor, "Unknown sensor")
        return (
            f"[MOCK] Elevated {top_sensor} ({desc}) readings suggest possible HPC seal degradation. "
            f"Recommend borescope inspection of high-pressure compressor at next scheduled maintenance."
        )
