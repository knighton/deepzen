from .. import backend as Z
from .metric import Metric


class Loss(Metric):
    def __call__(self, true, pred):
        raise NotImplementedError


class MeanSquaredError(Metric):
    def __call__(self, true, pred):
        return Z.sum(Z.pow(true - pred, 2))


class CategoricalCrossEntropy(Metric):
    def __call__(self, true, pred):
        pred = Z.clip(pred, 1e-6, 1 - 1e-6)
        x = -true * Z.log(pred)
        return Z.mean(x)


def unpack_loss(x, y_shapes):
    if isinstance(x, Loss):
        return x

    # TODO: bin/cat.

    klass = {
        'mean_squared_error': MeanSquaredError,
        'categorical_cross_entropy': CategoricalCrossEntropy,
    }[x]
    return klass()
