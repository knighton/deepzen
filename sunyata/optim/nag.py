from .. import api as Z
from .base.optimizer import Optimizer


class NAG(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        Optimizer.__init__(self)
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
        before = ctx.velocity
        ctx.velocity = ctx.momentum * ctx.velocity + ctx.lr * grad
        velocity = (1 + ctx.momentum) * ctx.velocity - ctx.momentum * before
        Z.assign_sub(data, velocity)
