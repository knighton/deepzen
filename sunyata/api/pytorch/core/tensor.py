import torch

from ...base.core.tensor import BaseTensorAPI


class PyTorchTensorAPI(BaseTensorAPI):
    def zeros(self, shape, dtype=None, device=None):
        x = torch.zeros(*shape)
        return self.cast(x, dtype, device)

    def ones(self, shape, dtype=None, device=None):
        x = torch.ones(*shape)
        return self.cast(x, dtype, device)

    def full(self, shape, value, dtype=None, device=None):
        x = value * torch.ones(*shape)
        return self.cast(x, dtype, device)

    def arange(self, begin, end, step=1, dtype=None, device=None):
        x = torch.arange(begin, end, step)
        return self.cast(x, dtype, device)

    def eye(self, dim, dtype=None, device=None):
        x = torch.eye(dim)
        return self.cast(x, dtype, device)

    def random_uniform(self, shape, min=0, max=1, dtype=None, device=None):
        x = torch.rand(*shape) * (max - min) + min
        return self.cast(x, dtype, device)

    def random_normal(self, shape, mean=0, std=1, dtype=None, device=None):
        x = torch.randn(*shape) * std + mean
        return self.cast(x, dtype, device)

    def random_binomial(self, shape, prob=0.5, dtype=None, device=None):
        x = torch.rand(*shape) <= prob
        return self.cast(x, dtype, device)
