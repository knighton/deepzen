class Metric(object):
    def __call__(self, true, pred):
        raise NotImplementedError


def collect_metrics(base_class, subclasses):
    name2class = {}
    for klass in subclasses:
        assert issubclass(klass, base_class)
        if isinstance(klass.name, str):
            names = [klass.name]
        elif isinstance(klass.name, tuple):
            names = klass.name
        else:
            assert False
        for name in names:
            assert name
            assert isinstance(name, str)
            assert name not in name2class
            name2class[name] = klass
    return name2class
