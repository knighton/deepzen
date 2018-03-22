from .base.transformer import Transformer


class Length(Transformer):
    def __init__(self, length, pad=None):
        self.length = length
        self.pad = pad

    def transform(self, xxx):
        rrr = []
        for xx in xxx:
            if len(xx) < self.length:
                rr = list(xx) + [self.pad] * (self.length - len(xx))
            else:
                rr = list(xx[:self.length])
            rrr.append(rr)
        return rrr

    def inverse_transform(self, xxx):
        rrr = []
        for xx in xxx:
            rr = []
            for i in range(self.length):
                x = xx[self.length - i - 1]
                if x != self.pad:
                    rr = xx[:-i]
                    break
            rrr.append(rr)
        return rrr
