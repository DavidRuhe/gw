import torch
from pyro.distributions import TransformModule
from torch.distributions import constraints


class PartialTransform(TransformModule):
    domain = constraints.real_vector
    codomain = constraints.real_vector

    def __init__(self, transform, dims_to_transform):
        super().__init__(cache_size=1)
        self.transform = transform
        self.dims_to_transform = dims_to_transform

    def _call(self, x):
        x_left, x_right = torch.split(
            x, [self.dims_to_transform, x.shape[1] - self.dims_to_transform], -1
        )

        x = x_left
        y = self.transform(x)
        log_abs_det_jacobian = self.transform.log_abs_det_jacobian(x, y)

        self._cached_log_abs_det_jacobian = log_abs_det_jacobian

        return torch.cat([y, x_right], -1)

    def _inverse(self, y):
        y_left, y_right = torch.split(
            y, [self.dims_to_transform, y.shape[1] - self.dims_to_transform], -1
        )

        y = y_left
        x = self.transform.inv(y)
        log_abs_det_jacobian = -self.transform.log_abs_det_jacobian(x, y)
        self._cached_log_abs_det_jacobian = log_abs_det_jacobian
        return torch.cat([x, y_right], -1)

    def clear_cache(self):
        self._cached_log_abs_det_jacobian = None

    def log_abs_det_jacobian(self, x, y):
        x_old, y_old = self._cached_x_y
        if not (x is x_old and y is y_old):
            raise NotImplementedError()

        if self._cached_log_abs_det_jacobian is None:
            raise NotImplementedError()

        log_abs_det_jacobian = self._cached_log_abs_det_jacobian
        self.clear_cache()
        return log_abs_det_jacobian
