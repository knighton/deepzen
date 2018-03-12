from ... import api as Z


class OptimizerContext(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Optimizer(object):
    def set_params(self, params):
        self.params = params
        self.contexts = \
            [OptimizerContext(**self.make_optim_context(x)) for x in params]

    def make_optim_context(self, param):
        raise NotImplementedError

    def step_one(self, data, grad, ctx):
        raise NotImplementedError

    def step(self):
        for param, context in zip(self.params, self.contexts):
            self.step_one(param, Z.grad(param), context)
