from pathlib import Path
import torch

from sklearn.decomposition import PCA
import numpy as np
import joblib

EMBEDDINGS_PATH = Path("train_embeddings.pt")
N_COMPONENETS = 251


print("Loading embeddings...")
embeddings = torch.load(EMBEDDINGS_PATH)
print("Embeddings ready.")

print("Preparing pca model...")
pca = PCA(n_components=0.95)
embeddigns = embeddings.numpy()

print("Reducing embeddings...")
X_train_pca = pca.fit_transform(embeddings)
print("Embeddings reduced.")

print(pca.n_components_)

# cum_var = np.cumsum(pca.explained_variance_ratio_)
# print(cum_var)

# n_components = np.argmax(cum_var >= 0.95) + 1
# print(n_components)

print("Saving reduced embeddings...")
np.save("train_embeddings_pca.npy", X_train_pca)
print("Reduced embeddngs saved.")

print("Saving pca model...")
joblib.dump(pca, "pca.pkl")
print("PCA model saved.")

# pca = joblib.load("pca.pkl")
# X_test_pca = pca.transform(test_embeddings.numpy())
