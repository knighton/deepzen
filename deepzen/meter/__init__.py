from .accuracy import *  # noqa
from .loss import *  # noqa


def unpack_meter(x, y_sample_shape):
    for unpack in [unpack_accuracy, unpack_loss]:
        meter = unpack(x, y_sample_shape)
        if meter:
            return meter
    return None
