from .. import api as Z
from .base.optimizer import Optimizer
from .base.registry import register_optimizer


@register_optimizer
class Adadelta(Optimizer):
    name = 'adadelta'

    def __init__(self, epsilon=1e-6, lr=1, rho=0.95):
        Optimizer.__init__(self)
        assert 0 < epsilon
        assert 0 < lr
        assert 0 < rho < 1
        self.epsilon = epsilon
        self.lr = lr
        self.rho = rho

    def make_optim_context(self, param):
        return {
            'cache': Z.zeros_like(param),
            'delta_cache': Z.zeros_like(param),
            'epsilon': self.epsilon,
            'lr': self.lr,
            'rho': self.rho,
        }

    def step_one(self, data, grad, ctx):
        ctx.cache = self.rho * ctx.cache + (1 - self.rho) * Z.square(grad)
        update = grad * Z.sqrt(ctx.delta_cache + ctx.epsilon) / \
            Z.sqrt(ctx.cache + ctx.epsilon)
        Z.assign_sub(data, ctx.lr * update)
        ctx.delta_cache = self.rho * ctx.delta_cache + \
            (1 - self.rho) * Z.square(update)
