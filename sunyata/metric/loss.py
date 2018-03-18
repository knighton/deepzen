from .. import api as Z
from ..util.dataset import is_sample_one_scalar
from .base.metric import Metric


class Loss(Metric):
    def __call__(self, true, pred):
        raise NotImplementedError


class MeanSquaredError(Loss):
    def __call__(self, true, pred):
        return Z.sum(Z.pow(true - pred, 2))


class CategoricalCrossEntropy(Loss):
    def __call__(self, true, pred):
        pred = Z.clip(pred, 1e-6, 1 - 1e-6)
        x = -true * Z.log(pred)
        return Z.mean(x)


def unpack_loss(loss, y_sample_shape):
    if isinstance(loss, Loss):
        return loss

    if loss in {'xe', 'cross_entropy'}:
        if is_sample_one_scalar(y_sample_shape):
            loss = 'binary_cross_entropy'
        else:
            loss = 'categorical_cross_entropy'

    klass = {
        'mean_squared_error': MeanSquaredError,
        'categorical_cross_entropy': CategoricalCrossEntropy,
    }[loss]
    return klass()
