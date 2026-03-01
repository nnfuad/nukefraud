from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def get_logistic_model():
    return LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        n_jobs=-1,
    )


def get_random_forest():
    return RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )