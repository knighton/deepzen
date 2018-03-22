from .. import api as Z
from .base.optimizer import Optimizer
from .base.registry import register_optimizer


@register_optimizer
class Adam(Optimizer):
    name = 'adam'

    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-6, lr=0.001):
        Optimizer.__init__(self)
        assert 0 < beta1 < 1
        assert 0 < beta2 < 1
        assert 0 < epsilon
        assert 0 < lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.lr = lr

    def make_optim_context(self, param):
        return {
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon,
            'lr': self.lr,
            'm': Z.zeros_like(param),
            'v': Z.zeros_like(param),
            't': 1,
        }

    def step_one(self, data, grad, ctx):
        ctx.m = ctx.beta1 * ctx.m + (1 - ctx.beta1) * grad
        ctx.v = ctx.beta2 * ctx.v + (1 - ctx.beta2) * Z.square(grad)
        beta2_t = (1 - ctx.beta2 ** ctx.t) ** 0.5
        beta1_t = 1 - ctx.beta1 ** ctx.t
        lr_t = ctx.lr * beta2_t / beta1_t
        Z.assign_sub(data, lr_t * ctx.m / (Z.sqrt(ctx.v) + ctx.epsilon))
        ctx.t += 1
