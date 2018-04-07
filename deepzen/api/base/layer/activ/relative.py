class BaseRelativeAPI(object):
    def log_softmax(self, x):
        return self.log(self.softmax(x))

    def log_softmin(self, x):
        return self.log(self.softmin(x))

    def softmax(self, x):
        axes = list(range(self.ndim(x)))[1:]
        e_x = self.exp(x)
        return e_x / self.sum(e_x, axes, True)

    def softmin(self, x):
        axes = list(range(self.ndim(x)))[1:]
        maxes = self.max(x, axes, True)
        e_x = self.exp(-maxes - x)
        return e_x / self.sum(e_x, axes, True)
