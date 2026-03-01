# app/api.py

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import torch
from pathlib import Path

from src.models.mlp import FraudMLP

app = FastAPI()

MODEL_DIR = Path("models")
DEVICE = torch.device("cpu")

# Load models once at startup
logistic = joblib.load(MODEL_DIR / "logistic_model.joblib")
logistic_scaler = joblib.load(MODEL_DIR / "scaler.joblib")

mlp_scaler = joblib.load(MODEL_DIR / "mlp_scaler.joblib")
mlp = FraudMLP(30)
mlp.load_state_dict(torch.load(MODEL_DIR / "mlp_model.pt", map_location=DEVICE))
mlp.eval()

THRESHOLD = 0.97


class Transaction(BaseModel):
    features: list
    model: str


@app.get("/")
def health_check():
    return {"status": "Nukefraud API running"}


@app.post("/predict")
def predict(transaction: Transaction):

    X = np.array(transaction.features).reshape(1, -1)

    if transaction.model == "Logistic Regression":
        X_scaled = logistic_scaler.transform(X)
        prob = logistic.predict_proba(X_scaled)[0][1]
    else:
        X_scaled = mlp_scaler.transform(X)
        tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            logits = mlp(tensor)
            prob = torch.sigmoid(logits).item()

    label = int(prob >= THRESHOLD)

    return {
        "fraud_probability": float(prob),
        "prediction": label,
        "threshold": THRESHOLD,
    }