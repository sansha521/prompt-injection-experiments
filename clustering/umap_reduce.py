# pip install umap-learn

import numpy as np
import umap
import joblib

# Load BAAI embeddings
X_train = np.load("train_baai_embeddings.npy")
print(X_train.shape)

# Fit UMAP
reducer = umap.UMAP(
    n_components=50,
    n_neighbors=15,
    min_dist=0.0,
    metric="cosine",
    random_state=42,
)

X_train_umap = reducer.fit_transform(X_train)

print(X_train_umap.shape)

# Save reduced embeddings
np.save("train_baai_umap50.npy", X_train_umap)

# Save the fitted UMAP model
joblib.dump(reducer, "umap50.joblib")

print("Done.")

# import numpy as np
# import joblib

# reducer = joblib.load("umap50.joblib")

# X_val = np.load("val_baai_embeddings.npy")
# X_val_umap = reducer.transform(X_val)

# np.save("val_baai_umap50.npy", X_val_umap)