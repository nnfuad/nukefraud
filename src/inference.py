# src/inference.py

import torch
import joblib
import numpy as np
from pathlib import Path

from .models.mlp import FraudMLP
from .config import MODEL_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# For resgression model performed better in case of cost-sensitive optimization
class FraudInference:

    def __init__(self):
        self.model = joblib.load(Path(MODEL_DIR) / "logistic_model.joblib")
        self.scaler = joblib.load(Path(MODEL_DIR) / "scaler.joblib")

        # Manually set the threshold based on the best threshold found during training
        self.threshold = 0.97

    def predict(self, input_array):

        scaled = self.scaler.transform(input_array)
        proba = self.model.predict_proba(scaled)[:, 1][0]

        label = int(proba >= self.threshold)

        return {
            "fraud_probability": float(proba),
            "prediction": label,
        }