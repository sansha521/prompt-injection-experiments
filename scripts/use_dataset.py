from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset():
    """Load the data from Hugging Face"""
    """Source: https://huggingface.co/datasets/wambosec/prompt-injections"""

    print(f"Loading dataset...")
    train_df = pd.read_parquet(
        "hf://datasets/wambosec/prompt-injections/data/train-00000-of-00001.parquet"
    )

    test_df = pd.read_parquet(
        "hf://datasets/wambosec/prompt-injections/data/test-00000-of-00001.parquet"
    )
    print(f"Train dataset of shape {train_df.shape} loaded.")
    print(f"Test dataset of shape {test_df.shape} loaded.\n")
    
    return train_df, test_df

def split_data(train_df):
    """Split train dataset into train and validation sets"""

    # split train dataset into train and validation sets using stratified sampling
    print(f"Splitting train data into train and validation sets...")

    X, y = train_df.drop(columns="label"), train_df["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    # combine X and y columns into single train and validation sets
    train = pd.concat([X_train, y_train], axis=1)    
    val = pd.concat([X_val, y_val], axis=1)

    print(f"Train set of shape {train.shape} created.")
    print(f"Validation set of shape {val.shape} created.\n")

    return train, val

def save_data(train, val, test_df):
    """Write the datasets to csv files"""

    print(f"Writing datasets to csv files...")

    # create datasets folder if not already existing
    out_dir = Path(__file__).resolve().parents[1] / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)

    # data file names and datasets
    datasets = {
        "train.csv": train, 
        "val.csv": val, 
        "test.csv": test_df
    }

    for data_name, dataset in datasets.items():
        path = out_dir / data_name
        dataset.to_csv(path, index=False)
        print(f"dataset file {data_name} created.")

    print(f"Train, val, and test csv files ready.\n")


def main():
    train_df, test_df = load_dataset()
    train, val = split_data(train_df)
    save_data(train, val, test_df)


if __name__ == "__main__":
    main()
