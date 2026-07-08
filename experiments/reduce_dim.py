from pathlib import Path
import argparse

import torch
import numpy as np
import joblib
from sklearn.decomposition import PCA


# EMBEDDINGS_PATH = Path("train_embeddings.pt")
# N_COMPONENETS = 251
PCA_MODEL_PATH = Path("pca.pkl")
DEFAULT_VARIANCE = 0.95

def fit_pca(embeddings_path, variance=DEFAULT_VARIANCE):
    print(f"Loading embeddings from {embeddings_path}...")
    embeddings = torch.load(embeddings_path).numpy()
    print("Embeddings ready.")

    print("Fitting PCA...")
    pca = PCA(n_components=variance)
    reduced = pca.fit_transform(embeddings)
    print(f"Reduced from {embeddings.shape[1]} to {pca.n_components_} dimensions.")

    # cum_var = np.cumsum(pca.explained_variance_ratio_)
    # print(cum_var)

    # n_components = np.argmax(cum_var >= 0.95) + 1
    # print(n_components)

    joblib.dump(pca, PCA_MODEL_PATH)

    output_path = embeddings_path.with_name(
        embeddings_path.stem + "_pca.npy"
    )
    np.save(output_path, reduced)

    print(f"Saved reduced embeddings to {output_path}")
    print(f"Saved PCA model to {PCA_MODEL_PATH}")


def apply_pca(embeddings_path):
    print(f"Loading PCA mode from {PCA_MODEL_PATH}")
    pca = joblib.load(PCA_MODEL_PATH)

    print(f"Loading embeddings from {embeddings_path}")
    embeddings = torch.load(embeddings_path).numpy()

    print("Applying PCA...")
    reduced = pca.transform(embeddings)

    output_path = embeddings_path.with_name(
        embeddings_path.stem + "_pca.npy"
    )
    np.save(output_path, reduced)

    print(f"Saved reduced emebddigns to {output_path}")


def main(split):
    embeddings_path = Path(f"{split}_embeddings.pt")

    if split == "train":
        print("Fitting PCA to train.")
        fit_pca(embeddings_path)
    else:
        print("Applying PCA.")
        apply_pca(embeddings_path)
    

if __name__ == "__main__":
    print("Inside __main__")
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
    )

    args = parser.parse_args()
    main(args.split)

### run : python pca_embeddings.py --split val / test