import os
import numpy as np
import torch
from tqdm import trange
from numpy.lib.format import open_memmap


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


def metropolis_hastings(
    initial_params,
    potential_fn,
    n_samples,
    initial_step_size=1,
    burn_in=1024,
    target_acceptance_rate=0.234,
    step_size_decay=0.999,
    print_interval=128,
    length_decision_history=512,
):
    """Run Metropolis-Hastings algorithm.
    Args:

        initial_params: tensor with initial parameters
        potential_fn: function that returns potential value to be maximized
        n_samples: number of samples to be generated
        initial_step_size: initial step size
        burn_in: number of samples to discard before starting to collect samples
        target_acceptance_rate: target acceptance rate
        step_size_decay: step size decay
        print_interval: number of samples between printing progress
        length_decision_history: length of decision history to keep

        Returns:
        samples: tensor of samples
    """
    n_events = len(initial_params)
    step_sizes = torch.ones(n_events) * initial_step_size

    assert torch.isfinite(potential_fn(initial_params)).all()

    x = initial_params

    samples = []
    decisions = []

    for i in trange(n_samples + burn_in + 1):
        if i > burn_in:
            samples.append(x.clone())

        x_ = x + torch.randn_like(x) * step_sizes

        ratios = potential_fn(x_) - potential_fn(x)
        u = torch.rand_like(ratios).log()
        accepts = u <= ratios
        decisions.append(accepts)
        x[accepts] = x_[accepts]
        accept_ratios = torch.mean(torch.stack(decisions).float(), dim=0)

        step_sizes[accept_ratios < target_acceptance_rate] *= step_size_decay
        step_sizes[accept_ratios >= target_acceptance_rate] /= step_size_decay

        decisions = decisions[-length_decision_history:]
        if i > burn_in and i % print_interval == 0:
            assert not torch.any(accept_ratios == 0), print(
                "Try increasing initial step size."
            )
            assert not torch.any(accept_ratios == 1)

    return torch.stack(samples)


def rand_between(shape, low, high):
    return torch.Tensor(*shape).uniform_(low, high)


class PowerPlusPeakSimulator:
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
        theta = prior.sample(self.num_events).squeeze()
        p_x_theta = torch.distributions.Normal(torch.sqrt(theta), self.sigmasq**0.5)
        data = p_x_theta.sample().squeeze()

        likelihood = GaussianLikelihood(
            mu_x_theta=lambda theta: torch.sqrt(theta), sigmasq=self.sigmasq**0.5
        )

        def potential_fn(theta):
            densities = torch.full((len(theta),), -torch.inf)
            mask = (theta > self.mmin) & (theta < self.mmax)
            theta = theta[mask]
            prior_log_prob = prior.log_prob(theta)
            likelihood_log_prob = likelihood.log_prob(data[mask], theta)
            densities[mask] = prior_log_prob + likelihood_log_prob
            return densities

        initial_params = rand_between((self.num_events,), self.mmin, self.mmax)

        theta_posterior = metropolis_hastings(
            initial_params=initial_params,
            potential_fn=potential_fn,
            n_samples=self.num_posterior_samples,
            burn_in=self.burn_in,
        )

        theta_posterior = theta_posterior.permute(
            1, 0
        )  # (num_events, num_posterior_samples)

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
            f"Simulated {self.num_events} events with {self.num_posterior_samples} posterior samples to {posterior_path}."
        )

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
