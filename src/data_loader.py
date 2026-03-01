import pandas as pd
from pathlib import Path


def load_raw_data(data_path: str = "data/raw/creditcard.csv") -> pd.DataFrame:
    """
    Load raw credit card fraud dataset.
    """
    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(path)
    return df