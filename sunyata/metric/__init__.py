from .accuracy import unpack_accuracy
from .loss import unpack_loss


def unpack_metric(x, y_sample_shape):
    for unpack_metric in [unpack_accuracy, unpack_loss]:
        metric = unpack_metric(x, y_sample_shape)
        if metric:
            return metric
    return None
