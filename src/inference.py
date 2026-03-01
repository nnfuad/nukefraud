import torch
import joblib
import numpy as np
from pathlib import Path

from .models.mlp import FraudMLP
from .config import MODEL_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FraudInference:

    def __init__(self):

        self.logistic = joblib.load(Path(MODEL_DIR) / "logistic_model.joblib")
        self.logistic_scaler = joblib.load(Path(MODEL_DIR) / "scaler.joblib")

        self.mlp_scaler = joblib.load(Path(MODEL_DIR) / "mlp_scaler.joblib")

        input_dim = 30
        self.mlp = FraudMLP(input_dim)
        self.mlp.load_state_dict(
            torch.load(Path(MODEL_DIR) / "mlp_model.pt", map_location=DEVICE)
        )
        self.mlp.to(DEVICE)
        self.mlp.eval()

        self.threshold = 0.97

    def predict(self, input_array, model_name):

        if model_name == "Logistic Regression":
            scaled = self.logistic_scaler.transform(input_array)
            proba = self.logistic.predict_proba(scaled)[:, 1][0]

        else:
            scaled = self.mlp_scaler.transform(input_array)
            tensor = torch.tensor(scaled, dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                logits = self.mlp(tensor)
                proba = torch.sigmoid(logits).cpu().numpy()[0][0]

        label = int(proba >= self.threshold)

        return {
            "fraud_probability": float(proba),
            "prediction": label,
            "threshold": self.threshold,
        }