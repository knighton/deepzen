import sys

from .base.distribution import Distribution
from .fixed import *  # noqa
from .random import *  # noqa


def unpack_distribution(x):
    if isinstance(x, Distribution):
        pass
    elif isinstance(x, str):
        module = sys.modules[__name__]
        x = getattr(module, x)()
    else:
        assert False
    return x
