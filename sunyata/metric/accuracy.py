from .. import api as Z
from ..util.dataset import is_sample_one_scalar
from .base.metric import collect_metrics, Metric


class Accuracy(Metric):
    pass


class BinaryAccuracy(Accuracy):
    name = 'binary_accuracy', 'bin_acc'
    def __call__(self, true, pred):
        return Z.binary_accuracy(true, pred)


class CategoricalAccuracy(Accuracy):
    name = 'categorical_accuracy', 'cat_acc'
    def __call__(self, true, pred):
        return Z.categorical_accuracy(true, pred)


class TopKAccuracy(Accuracy):
    @classmethod
    def unpack(cls, s):
        if s.startswith('accuracy@'):
            s = s[9:]
        elif s.startswith('acc@'):
            s = s[4:]
        else:
            return None

        try:
            k = int(s)
        except:
            return None

        return cls(k)

    def __init__(self, k):
        self._k = k

    def __call__(self, true, pred):
        return Z.top_k_accuracy(true, pred, self._k)


NAME2ACCURACY = collect_metrics(Accuracy, [BinaryAccuracy, CategoricalAccuracy])


def unpack_accuracy(x, y_sample_shape):
    if isinstance(x, Accuracy):
        return x

    if x in {'acc', 'accuracy'}:
        if is_sample_one_scalar(y_sample_shape):
            acc = BinaryAccuracy()
        else:
            acc = CategoricalAccuracy()
        return acc

    acc = NAME2ACCURACY.get(x)
    if acc is not None:
        return acc

    return TopKAccuracy.unpack(x)
