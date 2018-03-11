from .. import backend as Z
from .loss import *  # noqa


def categorical_accuracy(true, pred):
    true_indices = Z.argmax(true, -1)
    pred_indices = Z.argmax(pred, -1)
    hits = Z.equal(true_indices, pred_indices)
    hits = Z.cast(hits, 'float32')
    return hits.mean()
