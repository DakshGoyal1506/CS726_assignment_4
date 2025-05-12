import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from sklearn.manifold import TSNE
from tqdm import trange

from get_results import EnergyRegressor, FEAT_DIM, DEVICE

# Other settings
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
SEED = 42

# Set random seed for reproducibility
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

class Algo1_Sampler:
    def __init__(self, energy_net, tau, device=DEVICE):
        self.energy_net = energy_net.to(device)
        self.tau = tau
        self.device = device

    def sample(self, num_steps, burn_in, initial_sample):
        # Initialize
        X = initial_sample.to(self.device)
        X.requires_grad_(True)

        samples = []
        total_start_time = time.time()
        burnin_start_time = None

        for t in trange(num_steps):
            # Compute energy and gradient at current X
            energy = self.energy_net(X)
            grad = torch.autograd.grad(energy, X, create_graph=False)[0]

            # Propose new sample with Euler-Maruyama under no_grad to avoid growing graph
            noise = torch.randn_like(X)
            with torch.no_grad():
                X_dash = X - (self.tau / 2) * grad + math.sqrt(self.tau) * noise
            X_dash.requires_grad_(True)

            # Record burn-in time
            if t == 0:
                burnin_start_time = time.time()

            # Compute energy and gradient at proposed point for MH correction
            energy_dash = self.energy_net(X_dash)
            grad_dash = torch.autograd.grad(energy_dash, X_dash, create_graph=False)[0]

            # Metropolis-Hastings acceptance probability
            diff1 = X - (X_dash - (self.tau / 2) * grad_dash)
            diff2 = X_dash - (X - (self.tau / 2) * grad)
            log_q_rev = - diff1.pow(2).sum() / (4 * self.tau)
            log_q_for = - diff2.pow(2).sum() / (4 * self.tau)
            log_alpha = energy - energy_dash + log_q_rev - log_q_for
            alpha = torch.exp(log_alpha).clamp(max=1.0)

            # Accept or reject
            if torch.rand(1, device=self.device) < alpha:
                X_new = X_dash
            else:
                X_new = X

            # Detach to break computational graph
            X = X_new.detach().requires_grad_(True)

            # Capture burn-in time once
            if t == burn_in - 1 and burnin_start_time is not None:
                burnin_time = time.time() - burnin_start_time

            # Collect after burn-in
            if t >= burn_in:
                samples.append(X.clone())

        numpy_samples = torch.stack(samples).detach().cpu().numpy()
        total_time = time.time() - total_start_time
        return numpy_samples, total_time, burnin_time

class Algo2_Sampler:
    def __init__(self, energy_net, tau, device=DEVICE):
        self.energy_net = energy_net.to(device)
        self.tau = tau
        self.device = device

    def sample(self, num_steps, burn_in, initial_sample):
        X = initial_sample.to(self.device)
        X.requires_grad_(True)

        samples = []
        total_start_time = time.time()
        burnin_start_time = None

        for t in trange(num_steps):
            energy = self.energy_net(X)
            grad = torch.autograd.grad(energy, X, create_graph=False)[0]

            noise = torch.randn_like(X)
            with torch.no_grad():
                X_proposed = X - (self.tau / 2) * grad + math.sqrt(self.tau) * noise
            X = X_proposed.detach().requires_grad_(True)

            # Record burn-in time once
            if t == 0:
                burnin_start_time = time.time()
            if t == burn_in - 1 and burnin_start_time is not None:
                burnin_time = time.time() - burnin_start_time

            # Collect after burn-in
            if t >= burn_in:
                samples.append(X.clone())

        numpy_samples = torch.stack(samples).detach().cpu().numpy()
        total_time = time.time() - total_start_time
        return numpy_samples, total_time, burnin_time

if __name__ == "__main__":
    samples = 10000
    burn_in = 2500
    tau = 0.001
    num_steps = samples + burn_in

    energy_net = EnergyRegressor(FEAT_DIM).to(DEVICE)
    initial_sample = torch.randn(FEAT_DIM, device=DEVICE)

    # Initialize samplers
    sampler1 = Algo1_Sampler(energy_net, tau)
    sampler2 = Algo2_Sampler(energy_net, tau)

    # Run Algo 1
    samples1, time1, burnin1 = sampler1.sample(
        num_steps=num_steps,
        burn_in=burn_in,
        initial_sample=initial_sample
    )
    print(f"Algo1 sampling time: {time1:.4f} s, burn-in time: {burnin1:.4f} s, collected {samples1.shape[0]} samples.")

    # Run Algo 2
    samples2, time2, burnin2 = sampler2.sample(
        num_steps=num_steps,
        burn_in=burn_in,
        initial_sample=initial_sample
    )
    print(f"Algo2 sampling time: {time2:.4f} s, burn-in time: {burnin2:.4f} s, collected {samples2.shape[0]} samples.")

    # t-SNE visualization
    tsne = TSNE(n_components=2, random_state=SEED)

    # Plot for Algo1
    tsne_results1 = tsne.fit_transform(samples1)
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_results1[:, 0], tsne_results1[:, 1], alpha=0.5, label='Algo1')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Visualization of Algo1 Samples')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'tsne_algo1_{samples}.png')

    # Plot for Algo2
    tsne_results2 = tsne.fit_transform(samples2)
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_results2[:, 0], tsne_results2[:, 1], alpha=0.5, label='Algo2')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Visualization of Algo2 Samples')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'tsne_algo2_{samples}.png')

    # Combined visualization
    plt.figure(figsize=(12, 10))
    plt.scatter(tsne_results1[:, 0], tsne_results1[:, 1], alpha=0.5, label='Algo1')
    plt.scatter(tsne_results2[:, 0], tsne_results2[:, 1], alpha=0.5, label='Algo2')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('Comparison of Sampling Algorithms in t-SNE Space')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'tsne_combined_{samples}.png')
