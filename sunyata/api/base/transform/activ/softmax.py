class BaseSoftmaxAPI(object):
    def softmax(self, x):
        axes = list(range(self.ndim(x)))[1:]
        e_x = self.exp(x)
        return e_x / self.sum(e_x, axes, True)
