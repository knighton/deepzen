class BaseEpsilonAPI(object):
    def __init__(self, epsilon):
        self.set_epsilon(epsilon)

    def set_epsilon(self, epsilon):
        assert isinstance(epsilon, float)
        assert 0 < epsilon < 1e-2
        self._epsilon = epsilon

    def epsilon(self):
        return self._epsilon
