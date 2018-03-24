from tqdm import tqdm

from .base.transformer import Transformer


class Filter(Transformer):
    def __init__(self, filter_):
        self.filter = filter_

    def transform(self, x, verbose=0, depth=0):
        rrr = []
        if verbose == 2:
            x = tqdm(x, leave=False)
        for line in x:
            rr = list(filter(lambda token: token in self.filter, line))
            rrr.append(rr)
        return rrr

    def inverse_transform(x):
        return x
