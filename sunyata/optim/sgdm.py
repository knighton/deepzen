from .. import backend as Z
from .base.optimizer import Optimizer


def set_with_momentum(momentum, old_value, new_value):
    return momentum * old_value + (1 - momentum) * new_value


class SGDM(Optimizer):
    def __init__(self, lr=0.05, momentum=0.9):
        super().__init__()
        assert 0 < lr
        assert 0 <= momentum <= 1
        self.lr = lr
        self.momentum = momentum

    def make_optim_context(self, param):
        return {
            'lr': self.lr,
            'momentum': self.momentum,
            'velocity': Z.zeros_like(param),
        }

    def step_one(self, data, grad, ctx):
        self.velocity = set_with_momentum(
            ctx.momentum, ctx.velocity, ctx.lr * grad)
        Z.assign_sub(data, self.velocity)
