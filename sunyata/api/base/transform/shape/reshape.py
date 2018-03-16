class BaseReshapeAPI(object):
    def batch_reshape(self, x, batch_shape):
        batch_size = self.shape(x)[0]
        return self.reshape(x, (batch_size,) + batch_shape)

    def batch_flatten(self, x):
        return self.batch_reshape(x, (-1,))
