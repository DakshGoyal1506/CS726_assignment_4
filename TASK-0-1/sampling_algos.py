import os
import torch
import numpy as np
import matplotlib.pyplot as plt
# You can import any other torch modules you need below #

import time
import math
from sklearn.manifold import TSNE

from get_results import EnergyRegressor, FEAT_DIM, DEVICE

##########################################################

# Other settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# Set random seed for reproducibility
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- Define two classes for Algo-1 and Algo-2 ---
##################################################
# Your code for Task-1 goes here
class Algo1_Sampler:

    def __init__(self, energy_net, tau, device=DEVICE):
        self.energy_net = energy_net.to(device)
        self.tau = tau
        self.device = device
        pass
    def sample(self, num_steps, burn_in, initial_sample):
        X = initial_sample.to(DEVICE)
        X.requires_grad_(True)

        samples = []
        start_time = time.time()

        for t in range(num_steps):
            energy = self.energy_net(X)
            grad = torch.autograd.grad(energy, X, create_graph=True)[0]

            noise = torch.randn_like(X)
            X_dash = X - (self.tau / 2) * grad + math.sqrt(self.tau) * noise
            X_dash = X_dash.detach().clone().to(self.device)
            X_dash.requires_grad_(True)

            energy_dash = self.energy_net(X_dash)
            grad_dash = torch.autograd.grad(energy_dash, X_dash, create_graph=True)[0]

            diff1 = X - (X_dash - (self.tau / 2) * grad_dash)
            diff2 = X_dash - (X - (self.tau / 2) * grad)

            log_q_reverse = - torch.sum(diff1 ** 2) / (4 * self.tau)
            log_q_forward = - torch.sum(diff2 ** 2) / (4 * self.tau)

            log_alpha = (energy - energy_dash + log_q_reverse - log_q_forward)
            alpha = torch.exp(log_alpha).clamp(max=1.0)

            u = torch.rand(1).to(self.device)

            if u < alpha:
                X = X_dash
            
            if t >= burn_in:
                samples.append(X.detach().cpu().numpy())
        
        total_time = time.time() - start_time
        return np.array(samples), total_time
    
class Algo2_Sampler:

    def __init__(self, energy_net, tau, device=DEVICE):
        self.energy_net = energy_net.to(device)
        self.tau = tau
        self.device = device
        pass

    def sample(self, num_steps, burn_in, initial_sample):
        X = initial_sample.to(self.device)
        X.requires_grad_(True)

        samples = []
        start_time = time.time()

        for t in range(num_steps):

            energy = self.energy_net(X)
            grad = torch.autograd.grad(energy, X, create_graph=True)[0]

            noise = torch.randn_like(X)
            X = X - (self.tau / 2) * grad + math.sqrt(self.tau) * noise
            X = X.detach().clone().to(self.device)
            X.requires_grad_(True)

            if t >= burn_in:
                samples.append(X.detach().cpu().numpy())
        
        total_time = time.time() - start_time
        return np.array(samples), total_time

# --- Main Execution ---
if __name__ == "__main__":

    num_steps = 5000
    burn_in = 1000
    tau = 0.01

    energy_net = EnergyRegressor(FEAT_DIM).to(DEVICE)

    initial_sample = torch.randn(FEAT_DIM, device=DEVICE)

    sampler1 = Algo1_Sampler(energy_net, tau)
    sampler2 = Algo2_Sampler(energy_net, tau)

    samples1, time1 = sampler1.sample(num_steps=num_steps, burn_in=burn_in, initial_sample=initial_sample)
    print(f"Algo1 sampling time: {time1:.4f} seconds, collected {samples1.shape[0]} samples.")

    samples2, time2 = sampler2.sample(num_steps=num_steps, burn_in=burn_in, initial_sample=initial_sample)
    print(f"Algo2 sampling time: {time2:.4f} seconds, collected {samples2.shape[0]} samples.")

    combined_samples = np.vstack((samples1, samples2))

    tsne = TSNE(n_components=2, random_state=SEED)
    tsne_results = tsne.fit_transform(combined_samples)

    labels = np.array([0]*samples1.shape[0] + [1]*samples2.shape[0])

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Visualization of Samples')
    plt.colorbar(scatter, label='Sampler')
    plt.legend(*scatter.legend_elements(), title="Samplers")
    plt.show()