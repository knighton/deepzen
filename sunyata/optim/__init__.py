from .base.optimizer import Optimizer
from .nag import NAG
from .sgd import SGD
from .sgdm import SGDM


def unpack_optim(x):
    if isinstance(x, Optimizer):
        return x
    klass = {
        'nag': NAG,
        'sgd': SGD,
        'sgdm': SGDM,
    }[x]
    return klass()
