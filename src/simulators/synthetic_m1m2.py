import argparse
import os
import numpy as np
import torch
from numpy.lib.format import open_memmap


lpeak = 0.1
mmax = 86.22
mmin = 4.59
alpha = 2.63
sigmam = 5.69
mum = 33.07
sigmasq = 1


class Mixture(torch.distributions.Distribution):
    def __init__(self, distributions, weights) -> None:
        assert sum(weights) == 1
        self.distributions = distributions
        self.weights = weights

    def sample(self, n):
        assert isinstance(n, int)
        samples = []
        nums = []
        for i in range(len(self.weights) - 1):
            nums.append(int(self.weights[i] * n))
        nums.append(n - sum(nums))

        for i in range(len(self.distributions)):
            num = nums[i]
            samples.append(self.distributions[i].sample((num,)))

        samples = torch.cat(samples, dim=0)
        samples = samples[torch.randperm(len(samples))]
        return samples

    def log_prob(self, x):

        log_probs = torch.stack(
            [self.distributions[i].log_prob(x) for i in range(len(self.distributions))]
        )
        log_prob = torch.logsumexp(log_probs, dim=0)
        return log_prob


class GaussianLikelihood:
    def __init__(self, mu_x_theta, sigmasq) -> None:
        self.mu_x_theta = mu_x_theta
        self.sigmasq = sigmasq

    def log_prob(self, x, theta):
        return torch.distributions.Normal(
            self.mu_x_theta(theta), self.sigmasq**0.5
        ).log_prob(x)


class PowerLaw(torch.distributions.Distribution):
    """p(x) = x^alpha, x > xmin"""

    def __init__(self, alpha, xmin):
        self.alpha = alpha
        self.xmin = xmin
        super().__init__(validate_args=False)

    def pdf(self, x):
        return (-self.alpha - 1) / self.xmin * (x / self.xmin) ** (self.alpha)

    def log_prob(self, x):
        return self.pdf(x).log()

    def sample(self, size):
        u = torch.rand(size)
        return self.icdf(u)

    def icdf(self, u):
        return self.xmin * (1 - u) ** (1 / (1 + self.alpha))


def rand_between(shape, low, high):
    return torch.Tensor(*shape).uniform_(low, high)


class SyntheticM1M2Simulator:
    def __init__(
        self,
        output_path,
        num_events,
        num_posterior_samples,
        lpeak=0.1,
        mmax=86.22,
        mmin=4.59,
        alpha=2.63,
        sigmam=5.69,
        mum=33.07,
        sigmasq=1,
        burn_in=1024,
    ) -> None:

        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.num_events = num_events
        self.num_posterior_samples = num_posterior_samples
        self.lpeak = lpeak
        self.mmax = mmax
        self.mmin = mmin
        self.alpha = alpha
        self.sigmam = sigmam
        self.mum = mum
        self.sigmasq = sigmasq
        self.burn_in = burn_in

    def run(self):
        # Generate events

        prior = Mixture(
            (
                torch.distributions.Normal(self.mum, self.sigmam),
                PowerLaw(-self.alpha, self.mmin),
            ),
            (self.lpeak, 1 - self.lpeak),
        )
        m1 = prior.sample(self.num_events).squeeze()
        m1 = m1[:, None] + 1e-4 * torch.randn(len(m1), self.num_posterior_samples)

        m1 = m1[torch.all(m1 > self.mmin, dim=1)]
        m1 = m1[torch.all(m1 < self.mmax, dim=1)]

        m2 = m1.clone()
        m2 = m1 - rand_between(m2.shape, 0, 4)
        m2 = np.clip(m2, self.mmin, None)
        m = torch.stack([m1, m2], dim=-1)

        marginal_path = os.path.join(self.output_path, "marginal.npy")
        memmap = open_memmap(
            marginal_path,
            mode="w+",
            dtype=np.float32,
            shape=(self.num_events,),
        )
        memmap[:] = theta.numpy()
        memmap.flush()
        del memmap
        print(f"Saved marginal samples to {marginal_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument("--num_events", type=int, default=1024)
    parser.add_argument("--num_posterior_samples", type=int, default=32768)
    args = parser.parse_args()

    simulator = SyntheticM1M2Simulator(
        args.output_path,
        args.num_events,
        args.num_posterior_samples,
    )
    simulator.run()
    print("Done.")
