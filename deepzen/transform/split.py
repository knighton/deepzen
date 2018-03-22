from .base.transformer import Transformer


class Split(Transformer):
    def transform(self, x):
        rrr = []
        for line in x:
            if isinstance(line, list):
                line = ''.join(line)
            tokens = line.split()
            rrr.append(tokens)
        return rrr

    def inverse_transform(x):
        rrr = []
        for line in x:
            rrr.append(' '.join(line))
        return rrr
