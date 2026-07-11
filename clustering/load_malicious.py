from pathlib import Path

import pandas as pd

def load_malicious():
    folder = Path(__file__).resolve().parents[1] / "data"

    train = pd.read_csv(folder / "train_malicious.csv")
    val = pd.read_csv(folder / "val_malicious.csv")
    test = pd.read_csv(folder / "test_malicious.csv")

    X_train, y_train = train["prompt"], train["label"]
    X_val, y_val, = val["prompt"], val["label"]
    X_test, y_test = test["prompt"], test["label"]

    return X_train, y_train, X_val, y_val, X_test, y_test