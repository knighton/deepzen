from .base.optimizer import Optimizer
from .adagrad import Adagrad
from .nag import NAG
from .sgd import SGD
from .sgdm import SGDM


def unpack_optim(x):
    if isinstance(x, Optimizer):
        return x
    klass = {
        'adagrad': Adagrad,
        'nag': NAG,
        'sgd': SGD,
        'sgdm': SGDM,
    }[x]
    return klass()
