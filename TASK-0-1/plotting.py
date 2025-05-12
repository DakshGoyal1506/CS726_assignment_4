import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Import the dataset from get_results.py
from get_results import EnergyDataset, DEVICE, SEED

# Path to the dataset file
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'A4_test_data.pt')

# Load the dataset
dataset = EnergyDataset(DATASET_PATH)
# Convert tensor data to NumPy arrays
x = dataset.x.numpy()           # shape: [N, 784]
energy = dataset.energy.squeeze().numpy()  # shape: [N]

# Limit dataset to at most 10,000 random datapoints
max_points = 10000
if x.shape[0] > max_points:
    np.random.seed(SEED)  # for reproducibility
    indices = np.random.choice(x.shape[0], size=max_points, replace=False)
    x = x[indices]
    energy = energy[indices]

# Run t-SNE to reduce dimensionality to 2D
tsne = TSNE(n_components=2, random_state=SEED)
tsne_results = tsne.fit_transform(x)

plt.figure(figsize=(10, 8))
# Use energy values for color mapping
scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1],
                      c=energy, cmap='viridis', alpha=0.5)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE Visualization of A4_test_data.pt')
plt.colorbar(scatter, label='Energy')

# Save the generated image
plt.savefig('tsne_visualization.png', dpi=300)
plt.show()