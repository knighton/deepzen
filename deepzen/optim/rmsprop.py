from .. import api as Z
from .base.optimizer import Optimizer
from .base.registry import register_optimizer


@register_optimizer
class RMSprop(Optimizer):
    name = 'rmsprop'

    def __init__(self, epsilon=1e-6, lr=0.001, rho=0.9):
        Optimizer.__init__(self)
        assert 0 < rho < 1
        assert 0 < epsilon
        assert 0 < lr
        self.epsilon = epsilon
        self.lr = lr
        self.rho = rho

    def make_optim_context(self, param):
        return {
            'cache': Z.zeros_like(param),
            'epsilon': self.epsilon,
            'lr': self.lr,
            'rho': self.rho,
        }

    def step_one(self, data, grad, ctx):
        ctx.cache = ctx.rho * ctx.cache + (1 - ctx.rho) * Z.square(grad)
        Z.assign_sub(data, ctx.lr * grad / (Z.sqrt(ctx.cache) + ctx.epsilon))
