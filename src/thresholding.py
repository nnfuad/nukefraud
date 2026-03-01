import numpy as np
from sklearn.metrics import confusion_matrix


def compute_cost(y_true, y_proba, threshold, fn_cost=1000, fp_cost=10):
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    total_cost = (fn * fn_cost) + (fp * fp_cost)

    return total_cost, tn, fp, fn, tp


def find_best_threshold(y_true, y_proba, fn_cost=1000, fp_cost=10):
    thresholds = np.linspace(0.01, 0.99, 100)

    best_threshold = 0.5
    lowest_cost = float("inf")

    for t in thresholds:
        cost, _, _, _, _ = compute_cost(y_true, y_proba, t, fn_cost, fp_cost)

        if cost < lowest_cost:
            lowest_cost = cost
            best_threshold = t

    return best_threshold, lowest_cost