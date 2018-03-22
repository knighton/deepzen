from .. import api as Z
from .base.optimizer import Optimizer
from .base.registry import register_optimizer


@register_optimizer
class SGDM(Optimizer):
    name = 'sgdm'

    def __init__(self, lr=0.01, momentum=0.9):
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
        ctx.velocity = self.momentum * ctx.velocity + ctx.lr * grad
        Z.assign_sub(data, ctx.velocity)
