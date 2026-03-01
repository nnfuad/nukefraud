import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split

from ..data_loader import load_raw_data
from ..preprocessing import split_data, scale_features
from ..evaluation import evaluate_model
from ..thresholding import find_best_threshold
from ..config import MODEL_DIR, RANDOM_STATE
from .mlp import FraudMLP


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Path(MODEL_DIR).mkdir(exist_ok=True) 

# Minibatch training+validation split+early stopping+threshold optimization for lessening overfittig.
def train_mlp():
    # Hyperparameters
    epochs = 50
    lr = 1e-3
    batch_size = 1024
    patience = 5
    
    df = load_raw_data()
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Create validation split
    X_train_scaled, X_val_scaled, y_train, y_val = train_test_split(
        X_train_scaled,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)

    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    input_dim = X_train_scaled.shape[1]
    model = FraudMLP(input_dim).to(DEVICE)

    # Compute positive class weight
    pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_model_path = Path(MODEL_DIR) / "best_mlp.pt"
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor.to(DEVICE))
            val_loss = criterion(val_outputs, y_val_tensor.to(DEVICE)).item()

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {avg_train_loss:.4f} "
            f"Val Loss: {val_loss:.4f}"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_path = Path(MODEL_DIR) / "best_mlp.pt"
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Load best model
    model.load_state_dict(torch.load(best_model_path))

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


    torch.save(model.state_dict(), f"{MODEL_DIR}/mlp_model.pt")
    joblib.dump(scaler, f"{MODEL_DIR}/mlp_scaler.joblib")

    print("\nImproved MLP model saved successfully.")


    # X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(DEVICE)
    # y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    # X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)

    # input_dim = X_train_scaled.shape[1]
    # model = FraudMLP(input_dim).to(DEVICE)

    
    # Compute positive class weight
    # pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(DEVICE)

    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # for epoch in range(epochs):
    #     model.train()

    #     optimizer.zero_grad()
    #     outputs = model(X_train_tensor)
    #     loss = criterion(outputs, y_train_tensor)
    #     loss.backward()
    #     optimizer.step()

    #     print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")

    # model.eval()
    # with torch.no_grad():
    #     logits = model(X_test_tensor)
    #     y_proba = torch.sigmoid(logits).cpu().numpy().flatten()

    # # Threshold optimization
    # best_threshold, best_cost = find_best_threshold(y_test, y_proba)

    # print("\n=== MLP Cost Optimization ===")
    # print("Best Threshold:", round(best_threshold, 4))
    # print("Lowest Expected Cost:", best_cost)

    # results = evaluate_model(y_test, y_proba, threshold=best_threshold)

    # print("\n=== MLP Evaluation @ Optimized Threshold ===")
    # print("ROC-AUC:", results["ROC_AUC"])
    # print("PR-AUC:", results["PR_AUC"])
    # print(results["Classification_Report"])
    # print("Confusion Matrix:\n", results["Confusion_Matrix"])

    # Path(MODEL_DIR).mkdir(exist_ok=True)

    # torch.save(model.state_dict(), f"{MODEL_DIR}/mlp_model.pt")
    # joblib.dump(scaler, f"{MODEL_DIR}/mlp_scaler.joblib")

    # print("\nMLP model saved successfully.")


if __name__ == "__main__":
    train_mlp()