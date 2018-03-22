from .base.transformer import Transformer


class EachSample(Transformer):
    def __init__(self, func, inverse=None):
        self.func = func
        self.inverse = inverse

    def transform(self, xx):
        rr = []
        for x in xx:
            r = self.func(x)
            rr.append(r)
        return rr

    def inverse_transform(self, xx):
        rr = []
        for x in xx:
            r = self.inverse(x)
            rr.append(r)
        return rr


class EachToken(Transformer):
    def __init__(self, func, inverse=None):
        self.func = func
        self.inverse = inverse

    def transform(self, xxx):
        rrr = []
        for xx in xxx:
            rr = []
            for x in xx:
                r = self.func(x)
                rr.append(r)
            rrr.append(rr)
        return rrr

    def inverse_transform(self, xxx):
        rrr = []
        for xx in xxx:
            rr = []
            for x in xx:
                r = self.inverse(x)
                rr.append(r)
            rrr.append(rr)
        return rrr
