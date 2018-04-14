from ..node import *  # noqa
from ..util.py import require_kwargs


def _build_vgg(block_confs, include_top=True, pooling=None, classes=1000):
    assert isinstance(include_top, bool)
    assert pooling in {None, 'avg', 'max'}
    assert isinstance(classes, int) and 1 <= classes

    steps = []
    for repeats, channels in block_confs:
        block = (Conv(channels) * repeats) > MaxPool 
        steps.append(block)

    if include_top:
        dense = lambda dim: Dense(dim) > ReLU
        tail = Flatten > dense(4096) * 2 > Dense(classes) > Softmax
    else:
        if pooling == 'avg':
            tail = GlobalAvgPool
        elif pooling == 'max':
            tail = GlobalMaxPool
        else:
            assert False
    steps.append(tail)

    return Sequence(*steps)


@require_kwargs
def VGG16(include_top=True, pooling=None, classes=1000):
    blocks = [(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)]
    return _build_vgg(blocks, include_top, pooling, classes)


@require_kwargs
def VGG19(include_top=True, pooling=None, classes=1000):
    blocks = [(2, 64), (2, 128), (4, 256), (4, 512), (4, 512)]
    return _build_vgg(blocks, include_top, pooling, classes)
