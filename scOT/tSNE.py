import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Example: Assume you have two datasets
# Dataset 1: 100 samples, 50 features
# Dataset 2: 150 samples, 50 features
# My dataset each: 10 samples, 1024*96 features

# Generate example datasets
np.random.seed(42)
jxf = np.load("/home/szehisaadat/sahba_work/poseidon/JXF.npy").mean(axis=1)
rp = np.load("/home/szehisaadat/sahba_work/poseidon/CE-RP.npy").mean(axis=1)
crp = np.load("/home/szehisaadat/sahba_work/poseidon/CE-CRP.npy").mean(axis=1)
kh = np.load("/home/szehisaadat/sahba_work/poseidon/CE-KH.npy").mean(axis=1)
ceg = np.load("/home/szehisaadat/sahba_work/poseidon/CE-Gauss.npy").mean(axis=1)
nsg = np.load("/home/szehisaadat/sahba_work/poseidon/NS-Gauss.npy").mean(axis=1)
nss = np.load("/home/szehisaadat/sahba_work/poseidon/NS-Gauss.npy").mean(axis=1)

# Combine datasets
data = np.vstack((jxf, rp, crp, kh, ceg, nsg, nss))

# Create labels (optional)
labels = np.array(["JXF"] * 100 + ["CE-RP"] * 100 + ["CE-CRP"] * 100 + ["CE-KH"] * 100 + ["CE-G"] * 100 + ["NS-G"] * 100 + ["NS-S"] * 100)  # 0 for Dataset 1, 1 for Dataset 2
# labels = np.array([0] * 100 + [1] * 100 + [2] * 100 + [3] * 100 + [4] * 100 + [5] * 100 + [6] * 100)  # 0 for Dataset 1, 1 for Dataset 2

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=5, learning_rate=200)
data_tsne = tsne.fit_transform(data_scaled)

# Plot the results
plt.figure(figsize=(10, 7))
for label in np.unique(labels):
    plt.scatter(data_tsne[labels == label, 0], data_tsne[labels == label, 1], 
                label=f"Dataset {label}", alpha=0.7)

plt.title("t-SNE Visualization")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend()
plt.savefig("tSNE-test.png")

exit()
