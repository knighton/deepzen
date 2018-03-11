from .base.optimizer import Optimizer
from .sgd import SGD
from .sgdm import SGDM


def unpack_optim(x):
    if isinstance(x, Optimizer):
        return x
    klass = {
        'sgd': SGD,
        'sgdm': SGDM,
    }[x]
    return klass()
