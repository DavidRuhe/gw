#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
import seaborn as sns
import matplotlib.pyplot as plt
import pyro.distributions as dist
import pyro.distributions.transforms as T


# In[2]:


d = 1

dataset = torch.cat([torch.randn(64, d) -3, torch.randn(64, d) + 3])


# In[3]:


import math
from functools import partial

import torch
import torch.nn as nn
from torch.distributions import Transform, constraints

from pyro.nn import DenseNN

from pyro.distributions.conditional import ConditionalTransformModule
from pyro.distributions.torch_transform import TransformModule


class ShiftScale(Transform):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, weights=None, biases=None, bound=None):
        super().__init__(cache_size=1)
        self.weights = weights
        self.biases = biases
        self.bound = bound

    def _call(self, x):
        a = self.weights() if callable(self.weights) else self.weights
        b = self.biases() if callable(self.biases) else self.biases
        return a * x + b

    def _inverse(self, y):

        a = self.weights() if callable(self.weights) else self.weights
        b = self.biases() if callable(self.biases) else self.biases

        return (y - b) / a

    def log_abs_det_jacobian(self, x, y):
        a = self.weights() if callable(self.weights) else self.weights
        ladj = a.abs().log()
        return ladj.squeeze()


class ShiftScaleTransform(ShiftScale, TransformModule):

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, input_dim, bound=None):
        super().__init__(bound=bound)

        self.weights = nn.Parameter(1 + 0.1 * torch.randn(input_dim, input_dim))
        self.biases = nn.Parameter(torch.randn(input_dim))
#         self.reset_parameters()


# In[4]:


def f(x, a):
    z =  torch.where((x > -1) & (x <= 1), a * x, x)
    z = torch.where(x > 1, x + a - 1, z)
    z = torch.where(x <= -1, x -a + 1, z)
    return z


# In[5]:


def f_i(y, a):
    z = torch.where(y > a, y - a + 1, y)
    z = torch.where(y <= -a, y + a - 1, z)
    z = torch.where((y <= a) & (y > -a), 1 / a * y, z)

    return z


# In[6]:


x = torch.linspace(-3, 3, 32)
plt.plot(f(x, 0.5))


# In[7]:


plt.plot(f_i(x, 0.5))


# In[8]:


torch.allclose(f_i(f(x, 0.5), 0.5), x)


# In[9]:


import torch.nn.functional as F


# In[10]:


math.log(2)


# In[11]:


class LeakyReLU(Transform):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, a=None, bound=None):
        super().__init__(cache_size=1)
        self.a = a
        self.bound = bound

    def _call(self, x):
        a = self.a() if callable(self.a) else self.a
        return x.where(x < 0, a.exp() * x).squeeze()
#         return f(x)

    def _inverse(self, y):
        a = self.a() if callable(self.a) else self.a
        return y.where(y < 0, 1 / a.exp() * y).squeeze()
#         return f_i(y)

    def log_abs_det_jacobian(self, x, y):
        a = self.a() if callable(self.a) else self.a
        return torch.where(
            (x >= 0.0), torch.zeros_like(x), torch.ones_like(x) * a
        ).squeeze()
#             a = self.a() if callable(self.a) else self.a
#         J = torch.where(
#             (x > -1) & (x <= 1), torch.ones_like(x) * math.log(2), torch.ones_like(x)
#         )
#         return J.squeeze()


class LeakyReLUTransform(LeakyReLU, TransformModule):

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, input_dim, bound=None):
        super().__init__(bound=bound)

        self.a = torch.tensor(0.)


# In[12]:


def leaky_relu(x, alpha):
    return torch.maximum(torch.tensor(0.), x) + alpha * torch.minimum(torch.tensor(0.), x)



# class LeakyReLUTransform(Transform):
#     domain = constraints.real_vector
#     codomain = constraints.real_vector
#     bijective = True
#     alpha = torch.tensor(0.)

#     def __init__(self, alpha=None, bound=None):
#         super().__init__(cache_size=1)
# #         self.alpha = alpha
#         self.bound = bound

#     def _call(self, x):
# #         return leaky_relu(x, self.alpha.exp())
#         return F.leaky_relu(x)

#     def _inverse(self, y):
#         return F.leaky_relu(x, negative_slope=100)
# #         return leaky_relu(y, 1 / self.alpha.exp())

#     def log_abs_det_jacobian(self, x, y):
#         return torch.where(
#             x >= 0.0, torch.zeros_like(x), torch.ones_like(x) * self.alpha
#         )

# class LeakyReLUTransform(torch.distributions.transforms.Transform):
#     r"""
#     Transform via the mapping :math:`y = \text{LeakyReLU}(x)`.
#     """
#     domain = constraints.real
#     codomain = constraints.positive
#     bijective = True
#     sign = +1
# #     alpha = 0.9
    

#     def _call(self, x):
#         return f(x, self.a.exp())
# #         return F.leaky_relu(x, negative_slope=self.alpha)

#     def _inverse(self, y):
#         return f_i(y, self.a.exp())
# #         return F.leaky_relu(y, negative_slope=1 / self.alpha)

#     def log_abs_det_jacobian(self, x, y):
#         J = torch.where((x >= -1) & (x < 1), torch.ones_like(x) * self.a, torch.zeros_like(x))
#         return J.squeeze()
# #         return torch.where(x >= 0., torch.zeros_like(x), torch.ones_like(x) * math.log(self.alpha))
    

# class LeakyReLUTransformModule(LeakyReLUTransform, TransformModule):

#     domain = constraints.real_vector
#     codomain = constraints.real_vector
#     bijective = True

#     def __init__(self):
#         super().__init__()
#         self.a = nn.Parameter(torch.zeros(1))

# #         self.a = torch.tensor(0.)


class PositivePowerTransform(Transform):
    r"""
    Transform via the mapping
    :math:`y=\operatorname{sign}(x)|x|^{\text{exponent}}`.

    Whereas :class:`~torch.distributions.transforms.PowerTransform` allows
    arbitrary ``exponent`` and restricts domain and codomain to postive values,
    this class restricts ``exponent > 0`` and allows real domain and codomain.

    .. warning:: The Jacobian is typically zero or infinite at the origin.
    """
    domain = constraints.real
    codomain = constraints.real
    bijective = True
    sign = +1

    def __init__(self, exponent, *, cache_size=0, validate_args=None):
        super().__init__(cache_size=cache_size)
#         if isinstance(exponent, int):
#             exponent = float(exponent)
#         exponent = torch.as_tensor(exponent)
#         self.exponent = exponent
    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return PositivePowerTransform(self.exponent, cache_size=cache_size)


#     def __eq__(self, other):
#         if not isinstance(other, PositivePowerTransform):
#             return False
#         return self.exponent.eq(other.exponent).all().item()

    def _call(self, x):
#         exponent = F.softplus(self.exponent)
        e = self.exponent.exp()
#         exponent = self.exponent.exp()
        return x.abs().pow(e) * x.sign()

    def _inverse(self, y):
#         exponent = self.exponent.exp()
#         exponent = F.softplus(self.exponent)
        e = self.exponent.exp()

        return y.abs().pow(e.reciprocal()) * y.sign()

    def log_abs_det_jacobian(self, x, y):
#         e = F.softplus(self.exponent)
        e = self.exponent.exp()

        return torch.squeeze(e.log() + (y / x).log())


    def forward_shape(self, shape):
#         e = F.softplus(self.exponent)
        e = self.exponent.exp()
        return torch.broadcast_shapes(shape, getattr(e, "shape", ()))


    def inverse_shape(self, shape):
#         e = F.softplus(self.exponent)
        e = self.exponent.exp()
        return torch.broadcast_shapes(shape, getattr(e, "shape", ()))


    
class PositivePowerTransformModule(PositivePowerTransform, TransformModule):

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self):
        super().__init__(None)
        self.exponent = nn.Parameter(torch.zeros(1))



# In[13]:


base_dist = dist.Normal(torch.zeros(d), torch.ones(d))
num_layers = 2
transform = []
for l in range(num_layers - 1):
#     transform.append(T.affine_coupling(1))
    transform.append(ShiftScaleTransform(d))
#     transform.append(MatrixExponential(d))
#     transform.append(T.SoftplusTransform())
#     transform.append(LeakyReLUTransform(d))
#     transform.append(LeakyReLUTransformModule())
    transform.append(PositivePowerTransformModule())
#     transform.append(T.ELUTransform())
transform.append(ShiftScaleTransform(d))
# transform.append(MatrixExponential(d))


transform_modules = nn.ModuleList([m for m in transform if isinstance(m, nn.Module)])
flow_dist = dist.TransformedDistribution(base_dist, transform)


# In[14]:


flow_dist.log_prob(torch.randn(2, 1)).shape


# In[15]:


flow_dist.sample((2,))


# In[16]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


count_parameters(transform_modules)


# In[17]:


x = torch.linspace(-5, 5, 32)


# In[18]:


y = flow_dist.sample((1024,))


# In[19]:


y


# In[20]:


sns.kdeplot(y.detach().squeeze())


# In[21]:


flow_dist.log_prob(x[:, None]).shape


# In[22]:


x = torch.linspace(-256, 256, 32768)


# In[23]:


plt.plot(x, flow_dist.log_prob(x[:, None]).exp().detach().numpy().squeeze())


# In[24]:


# matexp = MatrixExponential(d)


# In[25]:


from torch.distributions.utils import _sum_rightmost


def log_prob(transformed_dist, value):
        """
        Scores the sample by inverting the transform(s) and computing the score
        using the score of the base distribution and the log abs det jacobian.
        """
        if transformed_dist._validate_args:
            transformed_dist._validate_sample(value)
        event_dim = len(transformed_dist.event_shape)
        log_prob = 0.0
        y = value
        for transform in reversed(transformed_dist.transforms):
            x = transform.inv(y)
            assert torch.allclose(transform(x), y),  transform
            print(transform, x.mean())
            event_dim += transform.domain.event_dim - transform.codomain.event_dim
            log_prob = log_prob - _sum_rightmost(transform.log_abs_det_jacobian(x, y),
                                                 event_dim - transform.domain.event_dim)
            y = x

        log_prob = log_prob + _sum_rightmost(transformed_dist.base_dist.log_prob(y),
                                             event_dim - len(transformed_dist.base_dist.event_shape))
        return log_prob
    
log_probs = log_prob(flow_dist, dataset)


# In[26]:


# print(count_parameters(transform_modules))


# In[27]:


def log_prob(transformed_dist, value):
        """
        Scores the sample by inverting the transform(s) and computing the score
        using the score of the base distribution and the log abs det jacobian.
        """
        if transformed_dist._validate_args:
            transformed_dist._validate_sample(value)
        event_dim = len(transformed_dist.event_shape)
        log_prob = 0.0
        y = value
        for transform in reversed(transformed_dist.transforms):
            x = transform.inv(y)
            print(transform, x.mean())
            event_dim += transform.domain.event_dim - transform.codomain.event_dim
            log_prob = log_prob - _sum_rightmost(transform.log_abs_det_jacobian(x, y),
                                                 event_dim - transform.domain.event_dim)
            y = x

        log_prob = log_prob + _sum_rightmost(transformed_dist.base_dist.log_prob(y),
                                             event_dim - len(transformed_dist.base_dist.event_shape))
        return log_prob


# In[28]:


steps = 32768
optimizer = torch.optim.Adam(transform_modules.parameters(), lr=1e-3)
for step in range(steps+1):
    optimizer.zero_grad()
    loss = -flow_dist.log_prob(dataset).mean()
#     loss = -log_prob(flow_dist, dataset).mean()
    loss.backward()
    optimizer.step()
    flow_dist.clear_cache()
    
    
    
    if step % 500 == 0:
        print('step: {}, loss: {}'.format(step, loss.item()))

        sample = flow_dist.sample((1000,)).squeeze().numpy()
        sns.kdeplot(sample)
        plt.title(f"Mean: {sample.mean()}, STD: {sample.std()}")
        plt.show()


# In[29]:


with torch.no_grad():
    sample = flow_dist.sample((1000,)).squeeze().numpy()
    sns.kdeplot(sample)
    plt.title(f"Mean: {sample.mean()}, STD: {sample.std()}")


# In[34]:


x = torch.linspace(-8, 8, 128)


# In[35]:


plt.plot(x, flow_dist.log_prob(x[:, None]).exp().detach().numpy().squeeze())


# In[32]:


sns.kdeplot(dataset.squeeze())


# In[33]:


# 

