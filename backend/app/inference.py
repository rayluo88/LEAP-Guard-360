# backend/app/inference.py
import numpy as np
import joblib
import json
import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    """Must match the architecture used during training."""
    def __init__(self, n_features=8, hidden_dim=64, latent_dim=32, seq_len=10):
        super().__init__()
        self.seq_len = seq_len
        # Layer names must match the saved model weights
        self.encoder1 = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.encoder2 = nn.LSTM(hidden_dim, latent_dim, batch_first=True)
        self.decoder1 = nn.LSTM(latent_dim, latent_dim, batch_first=True)
        self.decoder2 = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        x, _ = self.encoder1(x)
        x, (hidden, _) = self.encoder2(x)
        latent = hidden.squeeze(0).unsqueeze(1).repeat(1, self.seq_len, 1)
        x, _ = self.decoder1(latent)
        x, _ = self.decoder2(x)
        return self.output_layer(x)


class AnomalyDetector:
    def __init__(self, model_dir: str = "model/"):
        with open(f"{model_dir}/config.json", "r") as f:
            self.config = json.load(f)

        # Actual config uses "input_sequence_length" and "n_features"
        n_features = self.config["n_features"]
        seq_len = self.config["input_sequence_length"]

        self.model = LSTMAutoencoder(
            n_features=n_features,
            seq_len=seq_len
        )
        self.model.load_state_dict(
            torch.load(f"{model_dir}/leap_guard_model.pth", map_location="cpu", weights_only=True)
        )
        self.model.eval()

        self.scaler = joblib.load(f"{model_dir}/leap_guard_scaler.pkl")

        # Actual config uses "threshold_mae" key
        with open(f"{model_dir}/threshold.json", "r") as f:
            threshold_data = json.load(f)
            self.default_threshold = threshold_data["threshold_mae"]

    def predict(self, sensor_readings: list, window_size: int, threshold: float = None):
        threshold = threshold or self.default_threshold

        X = np.array(sensor_readings)
        expected_len = self.config["input_sequence_length"]
        if window_size != expected_len:
            raise ValueError(
                f"window_size must match model sequence length ({expected_len}), got {window_size}"
            )
        expected_shape = (expected_len, self.config["n_features"])
        if X.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, got {X.shape}")

        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0)

        with torch.no_grad():
            X_reconstructed = self.model(X_tensor).numpy()

        mse_per_feature = np.mean((X_scaled - X_reconstructed[0]) ** 2, axis=0)
        anomaly_score = float(np.mean(mse_per_feature))

        # Get top 3 contributing sensors
        total_error = np.sum(mse_per_feature)
        if total_error == 0:
            total_error = 1e-10  # Avoid division by zero

        top_indices = np.argsort(mse_per_feature)[-3:][::-1]
        # Actual config uses "feature_columns" key
        feature_names = self.config["feature_columns"]
        contributions = {
            feature_names[i]: round(float(mse_per_feature[i] / total_error), 3)
            for i in top_indices
        }

        return {
            "anomaly_score": round(anomaly_score, 4),
            "is_anomaly": anomaly_score > threshold,
            "sensor_contributions": contributions
        }
