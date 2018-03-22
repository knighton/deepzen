from ...util.registry import Registry
from .hook import Hook


REGISTRY = Registry(Hook)


def register_hook(x):
    return REGISTRY.register(x)


def get_hook(x):
    return REGISTRY.get(x)
