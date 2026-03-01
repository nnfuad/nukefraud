import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import precision_recall_curve

from ..data_loader import load_raw_data
from ..preprocessing import split_data, scale_features
from ..thresholding import compute_cost
from ..models.mlp import FraudMLP


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compare_models():

    df = load_raw_data()
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Load Logistic Model
    logistic_model = joblib.load("models/logistic_model.joblib")
    y_proba_log = logistic_model.predict_proba(X_test_scaled)[:, 1]

    # Load MLP Model
    input_dim = X_train_scaled.shape[1]
    mlp_model = FraudMLP(input_dim)
    mlp_model.load_state_dict(torch.load("models/mlp_model.pt"))
    mlp_model.to(DEVICE)
    mlp_model.eval()

    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        logits = mlp_model(X_test_tensor)
        y_proba_mlp = torch.sigmoid(logits).cpu().numpy().flatten()

    # =========================
    # Precision–Recall Curves
    # =========================

    precision_log, recall_log, _ = precision_recall_curve(y_test, y_proba_log)
    precision_mlp, recall_mlp, _ = precision_recall_curve(y_test, y_proba_mlp)

    plt.figure()
    plt.plot(recall_log, precision_log)
    plt.plot(recall_mlp, precision_mlp)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.legend(["Logistic Regression", "MLP"])
    plt.savefig("results/figures/pr_curve_comparison.png")
    plt.close()

    # =========================
    # Cost vs Threshold Curves
    # =========================

    thresholds = np.linspace(0.01, 0.99, 100)

    costs_log = []
    costs_mlp = []

    for t in thresholds:
        cost_log, _, _, _, _ = compute_cost(y_test, y_proba_log, t)
        cost_mlp, _, _, _, _ = compute_cost(y_test, y_proba_mlp, t)

        costs_log.append(cost_log)
        costs_mlp.append(cost_mlp)

    plt.figure()
    plt.plot(thresholds, costs_log)
    plt.plot(thresholds, costs_mlp)
    plt.xlabel("Threshold")
    plt.ylabel("Expected Cost")
    plt.title("Cost vs Threshold")
    plt.legend(["Logistic Regression", "MLP"])
    plt.savefig("results/figures/cost_vs_threshold.png")
    plt.close()

    print("Comparison plots saved to results/figures/")


if __name__ == "__main__":
    compare_models()