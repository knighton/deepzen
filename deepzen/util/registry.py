from inspect import isclass


class Registry(object):
    def __init__(self, base_class):
        self._base_class = base_class
        self._name2get = {}
        self._parses = []

    @classmethod
    def _get_names(cls, x):
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

    @classmethod
    def _check_name(cls, name):
        assert name
        assert isinstance(name, str)
        for c in name:
            assert c.islower() or c == '_'

    def register(self, x):
        for name in self._get_names(x):
            self._check_name(name)
            assert name not in self._name2get
            self._name2get[name] = x
        if hasattr(x, 'parse'):
            self._parses.append(x.parse)
        return x

    def get(self, x):
        if isinstance(x, self._base_class):
            return x

        if isclass(x):
            assert issubclass(x, self._base_class)
            return x()

        get = self._name2get.get(x)
        if get is not None:
            return get()

        for parse in self._parses:
            obj = parse(x)
            if obj is not None:
                return obj

        assert False
