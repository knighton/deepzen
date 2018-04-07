class BaseLossAPI(object):
    def binary_cross_entropy(self, true, pred):
        pred = self.clip(pred, self.epsilon(), 1 - self.epsilon())
        x = -true * self.log(pred) - (1 - true) * self.log(1 - pred)
        return self.mean(x, -1)

    def categorical_cross_entropy(self, true, pred):
        pred = self.clip(pred, self.epsilon(), 1 - self.epsilon())
        x = -true * self.log(pred)
        return self.mean(x, -1)

    def categorical_hinge(self, true, pred):
        pos = self.sum(true * pred, -1)
        neg = self.max((1 - true) * pred, -1)
        return self.maximum(0, neg - pos + 1)

    def hinge(self, true, pred):
        x = self.maximum(1 - true * pred, 0)
        return self.mean(x, -1)

    def kullback_leibler_divergence(self, true, pred):
        true = self.clip(true, self.epsilon(), 1)
        pred = self.clip(pred, self.epsilon(), 1)
        x = true * self.log(true / pred)
        return self.mean(x, -1)

    def logcosh(self, true, pred):
        x = self.cosh(pred - true)
        x = self.log(x)
        return self.mean(x, -1)

    def mean_absolute_error(self, true, pred):
        x = self.abs(true - pred)
        return self.mean(x, -1)

    def mean_absolute_percentage_error(self, true, pred):
        x = (true - pred) / self.clip(self.abs(true), self.epsilon())
        return self.mean(self.abs(x), -1)

    def mean_squared_error(self, true, pred):
        x = self.square(true - pred)
        return self.mean(x, -1)

    def mean_squared_logarithmic_error(self, true, pred):
        log_pred = self.log1p(self.clip(pred, self.epsilon()))
        log_true = self.log1p(self.clip(true, self.epsilon()))
        x = self.square(log_pred - log_true)
        return self.mean(x, -1)

    def poisson(self, true, pred):
        pred = self.clip(pred, self.epsilon())
        x = pred - true * self.log(pred)
        return self.mean(x, -1)

    def squared_hinge(self, true, pred):
        x = self.square(self.maximum(1 - true * pred, 0))
        return self.mean(x, -1)
