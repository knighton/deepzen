from .. import backend as Z
from .base.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, lr=0.05):
        super().__init__()
        assert 0 < lr
        self.lr = lr

    def make_env(self, param):
        return {
            'data': param,
            'lr': self.lr,
        }

    def step_one(self, env):
        Z.assign_sub(env.data, env.lr * env.grad())
