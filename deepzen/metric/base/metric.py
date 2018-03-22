class Metric(object):
    def __call__(self, true, pred):
        raise NotImplementedError
