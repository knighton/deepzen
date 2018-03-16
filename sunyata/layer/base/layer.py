from ... import api as Z


class Layer(object):
    def __init__(self):
        self._params = []

    def param(self, x, learned=True):
        if learned:
            x = Z.variable(x)
            self._params.append(x)
        else:
            x = Z.constant(x)
        return x

    def params(self):
        return self._params

    def forward(self, x, is_training):
        raise NotImplementedError
