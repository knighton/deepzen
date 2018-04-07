from .. import api as Z
from ..util.dataset import is_sample_one_scalar
from ..util.registry import Registry
from .base.meter import Meter


class Accuracy(Meter):
    pass


REGISTRY = Registry(Accuracy)


def register_accuracy(x):
    return REGISTRY.register(x)


def unpack_accuracy(x, y_sample_shape):
    if x in {'accuracy', 'acc'}:
        if is_sample_one_scalar(y_sample_shape):
            x = 'binary_accuracy'
        else:
            x = 'categorical_accuracy'
    return REGISTRY.get(x)


@register_accuracy
class BinaryAccuracy(Accuracy):
    name = 'binary_accuracy', 'bin_acc'

    def __call__(self, true, pred):
        return Z.binary_accuracy(true, pred)


@register_accuracy
class CategoricalAccuracy(Accuracy):
    name = 'categorical_accuracy', 'cat_acc'

    def __call__(self, true, pred):
        return Z.categorical_accuracy(true, pred)


@register_accuracy
class TopKAccuracy(Accuracy):
    name = 'top_k_accuracy', 'top_k_acc'

    @classmethod
    def parse(cls, s):
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
