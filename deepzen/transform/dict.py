from collections import Counter
import numpy as np

from .base.transformer import Transformer


class Dict(Transformer):
    def __init__(self, max_vocab_size=None, min_token_usage=5,
                 oov_token='<OOV>', oov_index=0):
        self.max_vocab_size = max_vocab_size
        self.min_token_usage = min_token_usage
        self.token2index = None
        self.tokens = None
        self.oov_token = oov_token
        self.oov_index = oov_index

    def fit(self, x):
        token2usage = Counter()
        for line in x:
            for token in line:
                token2usage[token] += 1
        usages_tokens = []
        for token, usage in token2usage.items():
            if self.min_token_usage is not None and \
                    usage < self.min_token_usage:
                continue
            usages_tokens.append((usage, token))
        usages_tokens.sort(reverse=True)
        if self.max_vocab_size is not None:
            usages_tokens = usages_tokens[:self.max_vocab_size]
        self.token2index = {}
        self.tokens = [self.oov_token]
        for i, (usage, token) in enumerate(usages_tokens):
            self.token2index[token] = i + 1
            self.tokens.append(token)

    def transform(self, x):
        rrr = []
        for line in x:
            rr = []
            for token in line:
                r = self.token2index.get(token, self.oov_index)
                rr.append(r)
            rrr.append(rr)
        return np.array(rrr)

    def inverse_transform(self, x):
        rrr = []
        for line in x:
            rr = []
            for token in line:
                r = self.tokens[token]
                rr.append(r)
            rrr.append(rr)
        return rrr
