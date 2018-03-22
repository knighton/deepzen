from .base.registry import get_hook
from .progress_bar import *  # noqa
from .rows import *  # noqa
from .server import *  # noqa


def get_hooks(x):
    if x is None:
        xx = []
    elif isinstance(x, str):
        xx = x.split(',')
    elif isinstance(x, (list, tuple)):
        xx = x
    else:
        assert False
    return list(map(get_hook, xx))
