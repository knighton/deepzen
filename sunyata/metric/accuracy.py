from .. import api as Z
from .base.metric import Metric


class CategoricalAccuracy(Metric):
    def __call__(self, true, pred):
        true_indices = Z.argmax(true, -1)
        pred_indices = Z.argmax(pred, -1)
        hits = Z.equal(true_indices, pred_indices)
        hits = Z.cast(hits, 'float32')
        return Z.mean(hits)
