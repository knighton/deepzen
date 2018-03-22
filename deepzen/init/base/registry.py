from inspect import isclass

from .initializer import Initializer


NAME2GET = {}
UNPACKS = []


def _names_of(x):
    if hasattr(x, 'name'):
        if isinstance(x.name, str):
            names = x.name,
        elif isinstance(x.name, tuple):
            names = x.name
        else:
            assert False
    elif hasattr(x, '__name__'):
        names = x.__name__,
    else:
        assert False
    return names


def _check_name(name):
    assert name
    assert isinstance(name, str)
    for c in name:
        assert c.islower() or c == '_'


def put(x):
    global NAME2GET
    for name in _names_of(x):
        _check_name(name)
        assert name not in NAME2GET
        NAME2GET[name] = x
    if hasattr(x, 'unpack'):
        global UNPACKS
        UNPACKS.append(x.unpack)


def register_initializer(x):
    put(x)
    return x


def get(x):
    if isinstance(x, Initializer):
        return x

    if isclass(x):
        assert issubclass(x, Initializer)
        return x()

    get = NAME2GET.get(x)
    if get is not None:
        return get()

    for unpack in UNPACKS:
        obj = unpack(x)
        if obj is not None:
            return obj

    assert False
