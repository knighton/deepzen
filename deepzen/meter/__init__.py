from .accuracy import *  # noqa
from .loss import *  # noqa


def unpack_meter(x, y_sample_shape):
    for unpack in [unpack_accuracy, unpack_loss]:
        meter = unpack(x, y_sample_shape)
        if meter:
            return meter
    return None


def unpack_loss_and_extra_meters(x, y_sample_shape):
    if isinstance(x, (list, tuple)):
        xx = x
    else:
        xx = [x]
    meters = []
    meters.append(unpack_loss(xx[0], y_sample_shape))
    for x in xx[1:]:
        meters.append(unpack_meter(x, y_sample_shape))
    return meters


def parse_meter_lists_str(s):
    ss = s.split(' ')
    return [s.split(',') for s in ss]


def unpack_meter_lists(x, y_sample_shapes):
    if isinstance(x, str):
        if ' ' in x or ',' in x:
            xxx = parse_meter_lists_str(x)
        else:
            xxx = [x]
    else:
        xxx = x
    meter_lists = []
    assert len(xxx) == len(y_sample_shapes)
    for xx, y_sample_shape in zip(xxx, y_sample_shapes):
        meters = unpack_loss_and_extra_meters(xx, y_sample_shape)
        meter_lists.append(meters)
    return meter_lists
