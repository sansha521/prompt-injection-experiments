from pathlib import Path

import pandas as pd

def load_data():
    folder = Path(__file__).resolve().parents[1] / "data"

    # required = {"prompt", "label"}

    # for name, df in [("train", train), ("val", val), ("test", test)]:
    #     missing = required - set(df.columns)
    #     if missing:
    #         raise ValueError(f"{name}.csv is missing columns: {missing}")

    train = pd.read_csv(folder / "train.csv")
    val = pd.read_csv(folder / "val.csv")
    test = pd.read_csv(folder / "test.csv")

    X_train, y_train = train["prompt"], train["label"]
    X_val, y_val = val["prompt"], val["label"]
    X_test, y_test = test["prompt"], test["label"]

    return X_train, y_train, X_val, y_val, X_test, y_test