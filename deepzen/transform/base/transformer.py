class Transformer(object):
    def fit(self, x):
        pass

    def transform(self, x):
        return x

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        raise NotImplementedError
