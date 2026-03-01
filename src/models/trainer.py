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
    
    from ..thresholding import find_best_threshold
    
    best_threshold, best_cost = find_best_threshold(y_test, y_proba)

    print("\n=== Cost-Based Optimization ===")
    print("Best Threshold:", best_threshold)
    print("Lowest Expected Cost:", best_cost)

    print("\n=== Evaluation @ Optimized Threshold ===")
    optimized_results = evaluate_model(
        y_test, y_proba, threshold=best_threshold
    )
    print("ROC-AUC:", optimized_results["ROC_AUC"])
    print("PR-AUC:", optimized_results["PR_AUC"])
    print(optimized_results["Classification_Report"])
    print("Confusion Matrix:\n", optimized_results["Confusion_Matrix"])
    
    # results = evaluate_model(y_test, y_proba)
    # # Default evaluation (threshold = 0.5)
    # print("\n=== Evaluation @ Default Threshold (0.5) ===")
    # default_results = evaluate_model(y_test, y_proba, threshold=0.5)
    # print("ROC-AUC:", default_results["ROC_AUC"])
    # print("PR-AUC:", default_results["PR_AUC"])
    # print(default_results["Classification_Report"])
    # print("Confusion Matrix:\n", default_results["Confusion_Matrix"])
    
    # print("=== Evaluation ===")
    # print("ROC-AUC:", results["ROC_AUC"])
    # print("PR-AUC:", results["PR_AUC"])
    # print(results["Classification_Report"])
    # print("Confusion Matrix:\n", results["Confusion_Matrix"])

    Path(MODEL_DIR).mkdir(exist_ok=True)

    joblib.dump(model, f"{MODEL_DIR}/logistic_model.joblib")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.joblib")
    print("\nModel and scaler saved successfully.")


if __name__ == "__main__":
    train_baseline()