# src/models/trainer.py

import joblib
from pathlib import Path

from ..data_loader import load_raw_data
from ..preprocessing import split_data, scale_features
from ..evaluation import evaluate_model
from .baseline import get_logistic_model
from ..config import MODEL_DIR


def train_baseline():
    df = load_raw_data()

    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    model = get_logistic_model()
    model.fit(X_train_scaled, y_train)

    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    results = evaluate_model(y_test, y_proba)

    print("=== Evaluation ===")
    print("ROC-AUC:", results["ROC_AUC"])
    print("PR-AUC:", results["PR_AUC"])
    print(results["Classification_Report"])
    print("Confusion Matrix:\n", results["Confusion_Matrix"])

    Path(MODEL_DIR).mkdir(exist_ok=True)

    joblib.dump(model, f"{MODEL_DIR}/logistic_model.joblib")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.joblib")


if __name__ == "__main__":
    train_baseline()