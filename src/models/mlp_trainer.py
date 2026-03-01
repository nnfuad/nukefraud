import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import joblib

from ..data_loader import load_raw_data
from ..preprocessing import split_data, scale_features
from ..evaluation import evaluate_model
from ..thresholding import find_best_threshold
from ..config import MODEL_DIR
from .mlp import FraudMLP


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_mlp(epochs=20, lr=1e-3):
    df = load_raw_data()

    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(DEVICE)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)

    input_dim = X_train_scaled.shape[1]
    model = FraudMLP(input_dim).to(DEVICE)

    # Compute positive class weight
    pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor)
        y_proba = torch.sigmoid(logits).cpu().numpy().flatten()

    # Threshold optimization
    best_threshold, best_cost = find_best_threshold(y_test, y_proba)

    print("\n=== MLP Cost Optimization ===")
    print("Best Threshold:", round(best_threshold, 4))
    print("Lowest Expected Cost:", best_cost)

    results = evaluate_model(y_test, y_proba, threshold=best_threshold)

    print("\n=== MLP Evaluation @ Optimized Threshold ===")
    print("ROC-AUC:", results["ROC_AUC"])
    print("PR-AUC:", results["PR_AUC"])
    print(results["Classification_Report"])
    print("Confusion Matrix:\n", results["Confusion_Matrix"])

    Path(MODEL_DIR).mkdir(exist_ok=True)

    torch.save(model.state_dict(), f"{MODEL_DIR}/mlp_model.pt")
    joblib.dump(scaler, f"{MODEL_DIR}/mlp_scaler.joblib")

    print("\nMLP model saved successfully.")


if __name__ == "__main__":
    train_mlp()