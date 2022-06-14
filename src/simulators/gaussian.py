import math
import os
import torch
from numpy.lib.format import open_memmap
import unittest
import numpy as np


class GaussianSimulator:
    def __init__(
        self,
        output_path,
        num_events=1024,
        num_posterior_samples=32768,
        dim=1,
        mu_theta=0,
        sigmasq_theta=1,
        sigmasq_x_theta=1e-4,
        F=math.pi,
        b=math.pi,
    ):
        super().__init__()
        self.output_path = output_path
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

        mu_x = self.F * self.mu_theta + self.b
        sigmasq_x = self.sigmasq_x_theta + self.F**2 * self.sigmasq_theta

        events = torch.randn(self.num_events, self.dim) * math.sqrt(sigmasq_x) + mu_x
        mu_theta_x = mu_theta_x(events)

        theta = mu_theta_x + torch.randn(
            self.num_events, self.num_posterior_samples
        ) * math.sqrt(sigmasq_theta_x)

        memmap = open_memmap(
            self.output_path,
            mode="w+",
            dtype=np.float32,
            shape=(self.num_events, self.num_posterior_samples),
        )
        memmap[:] = theta.numpy()
        memmap.flush()
        del memmap

        print(
            f"Simulated {self.num_events} events with {self.num_posterior_samples} posterior samples."
        )
        print(f"Saved to {self.output_path}")


class TestGaussianSimulator(unittest.TestCase):
    def test_simulate(self):
        output_path = "test_gaussian_simulator.npy"
        simulator = GaussianSimulator(
            output_path, num_events=16, num_posterior_samples=32
        )
        simulator.run()
        memmap = open_memmap(output_path, mode="r")
        self.assertEqual(memmap.shape, (16, 32))
        os.remove(output_path)


if __name__ == "__main__":
    unittest.main()
