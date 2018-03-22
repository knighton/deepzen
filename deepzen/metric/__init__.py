from .accuracy import get_accuracy
from .loss import get_loss


def get_metric(x, y_sample_shape):
    for get_metric in [get_accuracy, get_loss]:
        metric = get_metric(x, y_sample_shape)
        if metric:
            return metric
    return None
