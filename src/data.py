from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

ID_COL = "PassengerId"
TARGET = "Transported"


def load_data():
    return pd.read_csv(DATA_DIR / "train.csv"), pd.read_csv(DATA_DIR / "test.csv")
