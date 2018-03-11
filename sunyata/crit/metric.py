from .. import backend as Z


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


def unpack_metric(x, y_shapes):
    if isinstance(x, Metric):
        return x

    # TODO: bin/cat.

    klass = {
        'categorical_accuracy': CategoricalAccuracy,
    }[x]
    return klass()
