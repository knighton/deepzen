import sys

from .base.initializer import Initializer
from .fixed import *  # noqa
from .random import *  # noqa


def unpack_initializer(x):
    if isinstance(x, Initializer):
        pass
    elif isinstance(x, str):
        module = sys.modules[__name__]
        x = getattr(module, x)()
    else:
        assert False
    return x
