from .. import api as Z
from .base.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, lr=0.05):
        super().__init__()
        assert 0 < lr
        self.lr = lr

    def make_optim_context(self, param):
        return {
            'lr': self.lr,
        }

    def step_one(self, data, grad, ctx):
        Z.assign_sub(data, ctx.lr * grad)
