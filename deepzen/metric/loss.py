from .. import api as Z
from ..util.dataset import is_sample_one_scalar
from .base.metric import collect_metrics, Metric


class Loss(Metric):
    pass


class BinaryCrossEntropy(Loss):
    name = 'binary_cross_entropy', 'bin_xe'

    def __call__(self, true, pred):
        return Z.binary_cross_entropy(true, pred)


class CategoricalCrossEntropy(Loss):
    name = 'categorical_cross_entropy', 'cat_xe'

    def __call__(self, true, pred):
        return Z.categorical_cross_entropy(true, pred)


class CategoricalHinge(Loss):
    name = 'categorical_hinge', 'cat_hinge'

    def __call__(self, true, pred):
        return Z.categorical_hinge(true, pred)


class Hinge(Loss):
    name = 'hinge'

    def __call__(self, true, pred):
        return Z.hinge(true, pred)


class KullbackLeiblerDivergence(Loss):
    name = 'kullback_leibler_divergence', 'kl_divergence', 'kl_div'

    def __call__(self, true, pred):
        return Z.kullback_leibler_divergence(true, pred)


class LogCosh(Loss):
    name = 'log_cosh'

    def __call__(self, true, pred):
        return Z.log_cosh(true, pred)


class MeanAbsoluteError(Loss):
    name = 'mean_absolute_error', 'mae'

    def __call__(self, true, pred):
        return Z.mean_absolute_error(true, pred)


class MeanAbsolutePercentageError(Loss):
    name = 'mean_absolute_percentage_error', 'mape'

    def __call__(self, true, pred):
        return Z.mean_absolute_percentage_error(true, pred)


class MeanSquaredError(Loss):
    name = 'mean_squared_error', 'mse'

    def __call__(self, true, pred):
        return Z.mean_squared_error(true, pred)


class MeanSquaredLogarithmicError(Loss):
    name = 'mean_squared_logarithmic_error', 'msle'

    def __call__(self, true, pred):
        return Z.mean_squared_logarithmic_error(true, pred)


class Poisson(Loss):
    name = 'poisson'

    def __call__(self, true, pred):
        return Z.poisson(true, pred)


class SquaredHinge(Loss):
    name = 'squared_hinge'

    def __call__(self, true, pred):
        return Z.squared_hinge(true, pred)


NAME2LOSS = collect_metrics(Loss, [
    BinaryCrossEntropy, CategoricalCrossEntropy, CategoricalHinge, Hinge,
    KullbackLeiblerDivergence, LogCosh, MeanAbsoluteError,
    MeanAbsolutePercentageError, MeanSquaredError, MeanSquaredLogarithmicError,
    Poisson, SquaredHinge
])


def unpack_loss(x, y_sample_shape):
    if isinstance(x, Loss):
        return x

    if x in {'xe', 'cross_entropy'}:
        if is_sample_one_scalar(y_sample_shape):
            loss = BinaryCrossEntropy()
        else:
            loss = CategoricalCrossEntropy()
        return loss

    return NAME2LOSS.get(x)
