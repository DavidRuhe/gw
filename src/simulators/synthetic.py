import os
import math
import torch
from numpy.lib.format import open_memmap
import numpy as np


class SyntheticSimulator:
    def __init__(
        self,
        output_path,
        num_events=4096,
        num_posterior_samples=1024,
        sigmasq_theta_x=1e-2,
    ):
        super().__init__()
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        self.num_events = num_events
        self.num_posterior_samples = num_posterior_samples
        self.sigmasq_theta_x = sigmasq_theta_x

    def pdf(self, x, mu, sigmasq):
        return torch.exp(-((x - mu) ** 2) / (2 * sigmasq)) / torch.sqrt(
            2 * math.pi * sigmasq
        )

    def mu_theta_x(self, x):
        return x.abs().sqrt() * torch.sign(x)

    def run(self):

        x = torch.randn(self.num_events, 1)

        mu_theta_x = self.mu_theta_x(x)

        theta_x = mu_theta_x + torch.randn(self.num_events, 1) * math.sqrt(
            self.sigmasq_theta_x
        )

        posterior_path = os.path.join(self.output_path, "posterior.npy")
        memmap = open_memmap(
            posterior_path,
            mode="w+",
            dtype=np.float32,
            shape=(self.num_events, self.num_posterior_samples),
        )
        memmap[:] = theta_x.numpy()
        memmap.flush()
        del memmap
        print(
            f"Simulated {self.num_events} events with {self.num_posterior_samples} posterior samples."
        )
        print(f"Saved to {posterior_path}")
