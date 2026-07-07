from pathlib import Path

import pandas as pd

ROOT_PATH = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_PATH / "data"

train = pd.read_csv( DATA_PATH / "train.csv" )
val = pd.read_csv( DATA_PATH / "val.csv" )
test = pd.read_csv( DATA_PATH / "test.csv" )

train = train[train["label"] == 1]
val = val[val["label"] == 1]
test = test[test["label"] == 1]

train.to_csv(DATA_PATH / "train_malicious.csv", index=False)
val.to_csv(DATA_PATH / "val_malicious.csv", index=False)
test.to_csv(DATA_PATH / "test_malicious.csv", index=False)


print("Done.")