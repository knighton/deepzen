from ... import backend as Z


class Optimizee(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def grad(self):
        return Z.grad(self.data)


class Optimizer(object):
    def set_params(self, xx):
        self.envs = [Optimizee(**self.make_env(x)) for x in xx]

    def make_env(self, x):
        raise NotImplementedError

    def step_one(self, env):
        raise NotImplementedError

    def step(self):
        for env in self.envs:
            self.step_one(env)
