from .. import api as Z
from .base.optimizer import Optimizer
from .base.registry import register_optimizer


@register_optimizer
class Adagrad(Optimizer):
    name = 'adagrad'

    def __init__(self, lr=0.01, decay=0.99, epsilon=1e-6):
        Optimizer.__init__(self)
        assert 0 < lr
        assert 0 <= decay <= 1
        assert 0 < epsilon
        self.lr = lr
        self.decay = decay
        self.epsilon = epsilon

    def make_optim_context(self, param):
        return {
            'lr': self.lr,
            'decay': self.decay,
            'epsilon': self.epsilon,
            'cache': Z.zeros_like(param),
        }

    def step_one(self, data, grad, ctx):
        ctx.cache *= ctx.decay
        ctx.cache += Z.square(grad)
        Z.assign_sub(data, ctx.lr * grad / Z.sqrt(ctx.cache + ctx.epsilon))
