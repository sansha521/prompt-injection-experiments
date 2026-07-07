from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances
# from sklearn.metrics import silhouette_score
# import matplotlib.pyplot as plt

X = np.load("train_embeddings_pca.npy")

k = 5

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init="auto"
)

cluster_labels = kmeans.fit_predict(X)

# print(cluster_labels.shape) # (2464,)
# print(np.bincount(cluster_labels)) # [481 255 479 649 600]


# understanding, validating, and refining the clusters
## Find representative prompts (cluster centroids)

# DATA_PATH = Path(__file__).resolve().parents[1] / "data"

# df = pd.read_csv( DATA_PATH / "train_malicious.csv" )

# df["cluster"] = cluster_labels

# distances = cosine_distances(
#     X,
#     kmeans.cluster_centers_
# )

# for c in range(k):
#     closest = np.argsort(distances[:, c])[:5]

#     print("\nCLUSTER", c)

#     for idx in closest:
#         print(df.iloc[idx]["prompt"])


"""
CLUSTER 0 : Multilingual Instruction Hijacking / Cross-Lingual System Prompt Extraction
CLUSTER 1 : Encoded System Prompt Extraction (Hex/Unicode/Percent Encoding)
CLUSTER 2 : Encoded Payload Injection / Base64 Command Execution & Instruction Smuggling
CLUSTER 3 : Roleplay-Based Safety Boundary Probing / Academic Pretext Jailbreak
CLUSTER 4 : Fake System Message & Privilege Escalation Impersonation


C0 — Cross-Lingual Override Attacks
C1 — Obfuscated Instruction Injection
C2 — Encoded Payload & Execution Attacks
C3 — Social Engineering / Pretextual Jailbreaks
C4 — Authority Impersonation & Privilege Escalation
"""

# Visualize the clusters

### Predict clusters for new prompts
# new_embedding = embed(new_prompt)

# new_pca = pca.transform(new_embedding)

# new_cluster = kmeans.predict(new_pca)