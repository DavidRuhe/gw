from functools import partial
import math
from click import progressbar
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import unittest

# import pymc3 as pm
from pyro.distributions import TorchDistribution
from torch.distributions import constraints
import pyro
from pyro.infer.mcmc import MCMC, NUTS, HMC

# import torch.distributions as D
from torch.distributions.mixture_same_family import MixtureSameFamily

from pathos.multiprocessing import ProcessingPool as Pool

# from multiprocessing import Pool

PRINT_INTERVAL = 32


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


def run_mcmc(i, initial_params, potential_fn, potential_fn_kwargs):
    potential_fn = partial(potential_fn, **potential_fn_kwargs)
    nuts = NUTS(potential_fn=potential_fn)
    mcmc = MCMC(
        kernel=nuts,
        warmup_steps=256,
        initial_params=initial_params,
        num_samples=1024,
    )

    mcmc.run()
    return mcmc.get_samples()["theta"]
    # samples.append(mcmc.get_samples()["theta"])


def potential_fn_nonlinear_gaussian(z, prior, sigmasq_x_theta, data):
    theta = z["theta"]
    theta = torch.nn.functional.softplus(theta)

    prior_log_prob = -prior.log_prob(theta)
    nll = -torch.distributions.Normal(
        torch.tanh(theta), sigmasq_x_theta**0.5
    ).log_prob(data)
    return torch.mean(prior_log_prob + nll)


def sample_population(num_events):

    samples = []
    # for run in range(1):

    num_events = 16
    n1 = torch.distributions.Normal(loc=torch.tensor([12.0]), scale=torch.tensor([1.0]))
    n2 = torch.distributions.Normal(loc=torch.tensor([24.0]), scale=torch.tensor([1.0]))
    logn = torch.distributions.LogNormal(
        loc=torch.tensor([1.0]), scale=torch.tensor([1.3])
    )
    prior = Mixture([n1, n2, logn], [0.3, 0.3, 0.4])
    theta = prior.sample(num_events)
    p_x_theta = torch.distributions.Normal(torch.tanh(theta), torch.tensor([0.1]))
    data = p_x_theta.sample((num_events,))

    sigmasq_x_theta = 0.1**2

    initial_params = {
        "theta": (torch.FloatTensor(1, 1).uniform_(theta.min(), theta.max()))
    }
    potential_fn_kwargs = dict(prior=prior, sigmasq_x_theta=sigmasq_x_theta, data=data)

    kwargs = {
        "potential_fn": potential_fn_nonlinear_gaussian,
        "potential_fn_kwargs": potential_fn_kwargs,
        "initial_params": initial_params,
    }

    # Note! MCMC class num_chains also uses multiprocessing, but this approach ensures that I  can pass single events and run them in parallel (probably Pyro  can do this too, but not sure how)
    with Pool() as p:
        samples = p.map(
            partial(run_mcmc, **kwargs),
            range(4),
        )

    return samples


class GaussianLikelihood:
    def __init__(self, mu_x_theta, sigmasq) -> None:
        self.mu_x_theta = mu_x_theta
        self.sigmasq = sigmasq

    def log_prob(self, x, theta):
        return torch.distributions.Normal(
            self.mu_x_theta(theta), self.sigmasq**0.5
        ).log_prob(x)


def unnormalized_posterior(z, prior, likelihood, data):
    theta = z["theta"]
    # theta = torch.nn.functional.softplus(theta)
    theta = torch.exp(theta)

    prior_log_prob = -prior.log_prob(theta).exp()
    nll = -likelihood.log_prob(data, theta).exp()
    return torch.mean(prior_log_prob + nll)


def sample_posterior(i_datum, prior, likelihood, num_samples):

    i, datum = i_datum
    if i % PRINT_INTERVAL == 0:
        print(f"Starting sample {i}.")
    potential_fn = partial(
        unnormalized_posterior, prior=prior, likelihood=likelihood, data=datum
    )

    nuts = NUTS(potential_fn=potential_fn)
    mcmc = MCMC(
        kernel=nuts,
        warmup_steps=256,
        initial_params={"theta": 33.07 + torch.randn(1, 1)},
        num_samples=num_samples,
        disable_progbar=True,
    )
    mcmc.run()

    return mcmc.get_samples()["theta"]


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


if __name__ == "__main__":
    # samples = sample_population(128)
    # samples = torch.cat(samples, dim=0)
    # samples = samples.squeeze().numpy()
    # print(len(samples))
    # sns.kdeplot(samples)
    # plt.savefig("theta_samples.png")

    # num_events = 128
    # n1 = torch.distributions.Normal(loc=torch.tensor([12.0]), scale=torch.tensor([1.0]))
    # n2 = torch.distributions.Normal(loc=torch.tensor([24.0]), scale=torch.tensor([1.0]))
    # logn = torch.distributions.LogNormal(
    #     loc=torch.tensor([1.0]), scale=torch.tensor([1.3])
    # )
    # prior = Mixture([n1, n2, logn], [0.3, 0.3, 0.4])
    # theta = prior.sample(num_events).squeeze()

    likelihood = GaussianLikelihood(
        mu_x_theta=lambda theta: torch.tanh(theta), sigmasq=0.1**2
    )
    lpeak = 0.1
    mmax = 86.22
    mmin = 4.59
    alpha = 2.63
    sigmam = 5.69
    mum = 33.07
    prior = Mixture(
        (torch.distributions.Normal(mum, sigmam), PowerLaw(-alpha, mmin)),
        (lpeak, 1 - lpeak),
    )
    num_events = 1024
    theta = prior.sample(num_events).squeeze()
    p_x_theta = torch.distributions.Normal(torch.tanh(theta), torch.tensor([0.1]))
    data = p_x_theta.sample((1,)).squeeze()

    with Pool() as p:
        # for datum in data:
        #     thetas = sample_posterior(datum, prior, likelihood)
        map_fn = partial(
            sample_posterior, prior=prior, likelihood=likelihood, num_samples=1
        )
        thetas = p.map(map_fn, enumerate(data))

    thetas = torch.stack(thetas).squeeze(-1).squeeze(-1)
    # for i, theta in enumerate(thetas):
    #     sns.kdeplot(theta.numpy())
    #     plt.savefig(f"theta_posterior_{i}.png")
    #     plt.close()

    breakpoint()
    thetas_marginal = thetas[:, -1].exp()
    sns.kdeplot(thetas_marginal, label="mcmc marginal")
    sns.kdeplot(theta, label="true marginal")
    plt.legend()
    plt.savefig("theta_marginal.png")
    plt.close()
