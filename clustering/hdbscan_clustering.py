### baai_embeddings -> umap_reduce -> hdbscan_clustering

# pip install hdbscan

import numpy as np
import hdbscan

X_train = np.load("train_baai_umap50.npy")
print(len(X_train))

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=100,
    min_samples=5,
    metric="euclidean",
    prediction_data=True,
)

train_labels = clusterer.fit_predict(X_train)

print(f"Clusters: {len(set(train_labels)) - (1 if -1 in train_labels else 0)}")
print(f"Noise points: {(train_labels == -1).sum()}")

# Try different min_cluster_size
# for mcs in [20, 30, 50, 75, 100]:
#     clusterer = hdbscan.HDBSCAN(
#         min_cluster_size=mcs,
#         min_samples=5,
#         metric="euclidean",
#         prediction_data=True,
#     )
    
#     labels = clusterer.fit_predict(X_train)

#     n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
#     noise = np.sum(labels == -1)

#     print(
#         mcs,
#         "clusters:", n_clusters,
#         "noise:", noise,
#         "largest cluster:",
#         np.bincount(labels[labels>=0]).max()
#     )

# min_cluster_size = 100 : Clusters: 5, Noise points: 357

# cluster sizes
# import numpy as np

# labels = clusterer.labels_

unique, counts = np.unique(train_labels, return_counts=True)
print(dict(zip(unique, counts)))

