from ...util.registry import Registry
from .spy import Spy


REGISTRY = Registry(Spy)


def register_spy(x):
    return REGISTRY.register(x)


def unpack_spy(x):
    return REGISTRY.get(x)


def unpack_spies(x):
    if x is None:
        xx = []
    elif isinstance(x, str):
        xx = x.split(',')
    elif isinstance(x, (list, tuple)):
        xx = x
    else:
        assert False
    return list(map(unpack_spy, xx))
