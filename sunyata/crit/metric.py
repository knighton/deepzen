from .. import api as Z
from ..util.dataset import is_sample_one_scalar


class Metric(object):
    def __call__(self, true, pred):
        raise NotImplementedError


class CategoricalAccuracy(object):
    def __call__(self, true, pred):
        true_indices = Z.argmax(true, -1)
        pred_indices = Z.argmax(pred, -1)
        hits = Z.equal(true_indices, pred_indices)
        hits = Z.cast(hits, 'float32')
        return Z.mean(hits)


def unpack_metric(metric, y_sample_shape):
    if isinstance(metric, Metric):
        return metric

    if metric in {'acc', 'accuracy'}:
        if is_sample_one_scalar(y_sample_shape):
            metric = 'binary_accuracy'
        else:
            metric = 'categorical_accuracy'

    klass = {
        'categorical_accuracy': CategoricalAccuracy,
    }[metric]
    return klass()
