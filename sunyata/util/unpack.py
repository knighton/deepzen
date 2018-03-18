def unpack_dim(x):
    if isinstance(x, int):
        pass
    elif isinstance(x, (list, tuple)):
        assert len(x) == 1
        x, = x
    else:
        assert False
    return x


def unpack_shape(x, ndim):
    if isinstance(x, int):
        x = (x,) * ndim
    elif isinstance(x, (list, tuple)):
        assert len(x) == ndim
    else:
        assert False
    return x
