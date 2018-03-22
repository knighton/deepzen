import numpy as np

from ..base.initializer import Initializer
from .normal import Normal
from .truncated_normal import TruncatedNormal
from .uniform import Uniform


class VarianceScaler(Initializer):
    @classmethod
    def get_fans(cls, shape, meaning):
        if meaning == 'kernel':
            mul = int(np.prod(shape[2:]))
            fan_in = shape[1] * mul
            fan_out = shape[0] * mul
        else:
            assert False
        return fan_in, fan_out

    @classmethod
    def weight_fans(cls, fan_mode, fan_in, fan_out):
        if fan_mode == 'avg':
            x = (fan_in + fan_out) / 2
        elif fan_mode == 'in':
            x = fan_in
        elif fan_mode == 'out':
            x = fan_out
        else:
            assert False
        return x

    @classmethod
    def apply_dist(cls, shape, dtype, dist, weighted_fan, scale):
        scale /= weighted_fan
        if dist == 'normal':
            std = np.sqrt(scale)
            x = Normal.make(shape, dtype, 0, std)
        elif dist == 'truncated_normal':
            std = np.sqrt(scale)
            x = TruncatedNormal.make(shape, dtype, 0, std)
        elif dist == 'uniform':
            limit = np.sqrt(3 * scale)
            x = Uniform.make(shape, dtype, -limit, limit)
        else:
            assert False
        return x

    @classmethod
    def make(cls, shape, dtype, meaning, dist, fan_mode, scale):
        fan_in, fan_out = cls.get_fans(shape, meaning)
        weighted_fan = cls.weight_fans(fan_mode, fan_in, fan_out)
        return cls.apply_dist(shape, dtype, dist, weighted_fan, scale)

    def __init__(self, dist, fan, scale):
        self.dist = dist
        self.fan = fan
        self.scale = scale

    def __call__(self, shape, dtype, meaning=None):
        return self.make(shape, dtype, meaning, self.dist, self.fan, self.scale)


def variance_scaler(dist, fan, scale):
    return VarianceScaler(dist, fan, scale)


def glorot_normal():
    return VarianceScaler('normal', 'avg', 1)


def glorot_truncated_normal():
    return VarianceScaler('truncated_normal', 'avg', 1)


def glorot_uniform():
    return VarianceScaler('uniform', 'avg', 1)


def he_normal():
    return VarianceScaler('normal', 'in', 2)


def he_truncated_normal():
    return VarianceScaler('truncated_normal', 'in', 2)


def he_uniform():
    return VarianceScaler('uniform', 'in', 2)


def lecun_normal():
    return VarianceScaler('normal', 'in', 1)


def lecun_truncated_normal():
    return VarianceScaler('truncated_normal', 'in', 1)


def lecun_uniform():
    return VarianceScaler('uniform', 'in', 1)
