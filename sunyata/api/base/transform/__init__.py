class BaseTransformAPI(object):
    def softmax(self, x):
        axes = list(range(self.ndim(x)))[1:]
        e_x = self.exp(x)
        return e_x / self.sum(e_x, axes, True)

    def batch_reshape(self, x, batch_shape):
        batch_size = self.shape(x)[0]
        return self.reshape(x, (batch_size,) + batch_shape)

    def batch_flatten(self, x):
        return self.batch_reshape(x, (-1,))

    def dense(self, x, kernel, bias):
        raise NotImplementedError
