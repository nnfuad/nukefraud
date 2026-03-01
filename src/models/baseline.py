from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def get_logistic_model():
    return LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs", # Use 'lbfgs' for better convergence on larger datasets, lbgs allows for L2 regularization which can help with imbalanced data
        # n_jobs=-1 is removed for In modern sklearn versions, LogisticRegression does not support n_jobs, as it is not parallelized.
    )


def get_random_forest():
    return RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )