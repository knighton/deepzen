from ..util.dataset import is_sample_one_scalar
from .base.metric import Metric
from .accuracy import CategoricalAccuracy
from .loss import CategoricalCrossEntropy, MeanSquaredError


def unpack_metric(x, y_sample_shape):
    if isinstance(x, Metric):
        return x
    if x in {'acc', 'accuracy'}:
        if is_sample_one_scalar(y_sample_shape):
            x = 'binary_accuracy'
        else:
            x = 'categorical_accuracy'
    elif x in {'xe', 'cross_entropy'}:
        if is_sample_one_scalar(y_sample_shape):
            x = 'binary_cross_entropy'
        else:
            x = 'categorical_cross_entropy'
    klass = {
        'categorical_accuracy': CategoricalAccuracy,
        'mean_squared_error': MeanSquaredError,
        'categorical_cross_entropy': CategoricalCrossEntropy,
    }[x]
    return klass()
