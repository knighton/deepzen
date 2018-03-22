from ...util.registry import Registry
from .initializer import Initializer


REGISTRY = Registry(Initializer)


def register_initializer(x):
    return REGISTRY.register(x)


def get_initializer(x):
    return REGISTRY.get(x)
