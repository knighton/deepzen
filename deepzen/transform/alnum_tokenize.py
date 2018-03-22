import re

from .base.transformer import Transformer


class AlnumTokenize(Transformer):
    def __init__(self):
        self.pattern = re.compile('[\W_]+', re.UNICODE)

    def transform(self, xx):
        rr = []
        for x in xx:
            r = self.pattern.sub('', x).lower()
            rr.append(r)
        return rr

    def inverse(self, xxx):
        rr = []
        for xx in xxx:
            r = ' '.join(xx)
            rr.append(r)
        return rr
