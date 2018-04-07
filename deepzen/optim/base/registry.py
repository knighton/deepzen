from ...util.registry import Registry
from .optimizer import Optimizer


REGISTRY = Registry(Optimizer)


def register_optimizer(x):
    return REGISTRY.register(x)


def unpack_optimizer(x):
    return REGISTRY.get(x)
