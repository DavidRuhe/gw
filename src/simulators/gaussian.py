import os
import math
import torch
from numpy.lib.format import open_memmap
import numpy as np


class GaussianSimulator:
    def __init__(
        self,
        output_path,
        num_events=1024,
        num_posterior_samples=32768,
        dim=1,
        mu_theta=1,
        sigmasq_theta=1,
        sigmasq_x_theta=1e-4,
        F=math.pi,
        b=math.pi,
    ):
        super().__init__()
        self.output_path = output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.num_events = num_events
        self.num_posterior_samples = num_posterior_samples
        self.dim = dim
        self.mu_theta = mu_theta
        self.sigmasq_theta = sigmasq_theta
        self.sigmasq_x_theta = sigmasq_x_theta
        self.F = F
        self.b = b

    def run(self):
        mu_x_theta = lambda theta: self.F * theta + self.b

        sigmasq_theta_x = (
            1 / self.sigmasq_theta + self.F**2 / self.sigmasq_x_theta
        ) ** (-1)
        mu_theta_x = lambda x: sigmasq_theta_x * (
            self.F / self.sigmasq_x_theta * (x - self.b)
            + self.mu_theta / self.sigmasq_theta
        )

        theta = (
            torch.randn(self.num_events, self.dim) * self.sigmasq_theta**0.5
            + self.mu_theta
        )

        x = (
            mu_x_theta(theta)
            + torch.randn(self.num_events, self.dim) * self.sigmasq_x_theta**0.5
        )

        mu_theta_x = mu_theta_x(x)

        theta_posterior = mu_theta_x + torch.randn(
            self.num_events, self.num_posterior_samples
        ) * math.sqrt(sigmasq_theta_x)

        posterior_path = os.path.join(self.output_path, "posterior.npy")
        memmap = open_memmap(
            posterior_path,
            mode="w+",
            dtype=np.float32,
            shape=(self.num_events, self.num_posterior_samples),
        )
        memmap[:] = theta_posterior.numpy()
        memmap.flush()
        del memmap
        print(
            f"Simulated {self.num_events} events with {self.num_posterior_samples} posterior samples."
        )
        print(f"Saved to {posterior_path}")

        marginal_path = os.path.join(self.output_path, "marginal.npy")
        memmap = open_memmap(
            marginal_path,
            mode="w+",
            dtype=np.float32,
            shape=(self.num_events, 1),
        )
        memmap[:] = theta.numpy()
        memmap.flush()
        del memmap
        print(f"Saved marginals to {marginal_path}")
