import numpy as np
from scipy.stats import truncnorm

from ..base.distribution import Distribution


class TruncatedNormal(Distribution):
    @classmethod
    def make(cls, shape, dtype, mean=0, std=0.05, min_stds=-2, max_stds=2):
        dist = truncnorm(min_stds, max_stds)
        x = dist.rvs(np.prod(shape)).reshape(shape)
        x = x * std + mean
        return x.astype(dtype)

    def __init__(self, mean=0, std=0.05, min_stds=-2, max_stds=2):
        self.mean = mean
        self.std = std
        self.min_stds = min_stds
        self.max_stds = max_stds

    def __call__(self, shape, dtype, meaning=None):
        return self.make(self.mean, self.std, self.min_stds, self.max_stds,
                         shape, dtype)


def truncated_normal_within_stds(mean=0, std=0.05, min_stds=-2, max_stds=2):
    return TruncatedNormal(mean, std, min_stds, max_stds)


def truncated_normal_within_values(mean=0, std=0.05, min_value=-0.1,
                                   max_value=0.1):
    min_stds = (min_value - mean) / std
    max_stds = (max_value - mean) / std
    return TruncatedNormal(mean, std, min_stds, max_stds)
