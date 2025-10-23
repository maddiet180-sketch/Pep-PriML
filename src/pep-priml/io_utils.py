import pandas as pd
import os
import yaml

def load_data(file_path: str) -> pd.DataFrame:
    """Load simulation data from .dat or .csv files."""
    return pd.read_csv(file_path, sep=r"\s+")

def save_features(df: pd.DataFrame, out_path: str):
    """Save extracted features to a CSV file."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

def load_config(path: str = "src/pep_pri_ml/config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
