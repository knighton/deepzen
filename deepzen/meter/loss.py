from .. import api as Z
from ..util.dataset import is_sample_one_scalar
from ..util.registry import Registry
from .base.meter import Meter


class Loss(Meter):
    pass


REGISTRY = Registry(Loss)


def register_loss(x):
    return REGISTRY.register(x)


def unpack_loss(x, y_sample_shape):
    if x in {'cross_entropy', 'xe'}:
        if is_sample_one_scalar(y_sample_shape):
            x = 'binary_cross_entropy'
        else:
            x = 'categorical_cross_entropy'
    return REGISTRY.get(x)


@register_loss
class BinaryCrossEntropy(Loss):
    name = 'binary_cross_entropy', 'bin_xe'

    def __call__(self, true, pred):
        return Z.binary_cross_entropy(true, pred)


@register_loss
class CategoricalCrossEntropy(Loss):
    name = 'categorical_cross_entropy', 'cat_xe'

    def __call__(self, true, pred):
        return Z.categorical_cross_entropy(true, pred)


@register_loss
class CategoricalHinge(Loss):
    name = 'categorical_hinge', 'cat_hinge'

    def __call__(self, true, pred):
        return Z.categorical_hinge(true, pred)


@register_loss
class Hinge(Loss):
    name = 'hinge'

    def __call__(self, true, pred):
        return Z.hinge(true, pred)


@register_loss
class KullbackLeiblerDivergence(Loss):
    name = 'kullback_leibler_divergence', 'kl_divergence', 'kl_div'

    def __call__(self, true, pred):
        return Z.kullback_leibler_divergence(true, pred)


@register_loss
class LogCosh(Loss):
    name = 'log_cosh'

    def __call__(self, true, pred):
        return Z.log_cosh(true, pred)


@register_loss
class MeanAbsoluteError(Loss):
    name = 'mean_absolute_error', 'mae'

    def __call__(self, true, pred):
        return Z.mean_absolute_error(true, pred)


@register_loss
class MeanAbsolutePercentageError(Loss):
    name = 'mean_absolute_percentage_error', 'mape'

    def __call__(self, true, pred):
        return Z.mean_absolute_percentage_error(true, pred)


@register_loss
class MeanSquaredError(Loss):
    name = 'mean_squared_error', 'mse'

    def __call__(self, true, pred):
        return Z.mean_squared_error(true, pred)


@register_loss
class MeanSquaredLogarithmicError(Loss):
    name = 'mean_squared_logarithmic_error', 'msle'

    def __call__(self, true, pred):
        return Z.mean_squared_logarithmic_error(true, pred)


@register_loss
class Poisson(Loss):
    name = 'poisson'

    def __call__(self, true, pred):
        return Z.poisson(true, pred)


@register_loss
class SquaredHinge(Loss):
    name = 'squared_hinge'

    def __call__(self, true, pred):
        return Z.squared_hinge(true, pred)
