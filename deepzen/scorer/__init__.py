from .accuracy import *  # noqa
from .loss import *  # noqa


def get_scorer(x, y_sample_shape):
    for get_scorer in [get_accuracy_scorer, get_loss_scorer]:
        scorer = get_scorer(x, y_sample_shape)
        if scorer:
            return scorer
    return None
