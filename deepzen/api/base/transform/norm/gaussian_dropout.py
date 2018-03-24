import numpy as np


class BaseGaussianDropoutAPI(object):
    def gaussian_dropout(self, x, is_training, rate=0.5, axis=None, space=None):
        if not is_training:
            return x
        if space is not None:
            assert self.spatial_ndim(x) == space
        mul_shape = self._dropout_mask_shape(self.shape(x), axis)
        std = np.sqrt(rate / (1 - rate))
        mul = self.random_normal(mul_shape, 1, std)
        return x * self.constant(mul)

    def gaussian_dropout0d(self, x, is_training, rate=0.5, axis=None):
        return self.gaussian_dropout(x, is_training, rate, axis, 0)

    def gaussian_dropout1d(self, x, is_training, rate=0.5, axis=None):
        return self.gaussian_dropout(x, is_training, rate, axis, 1)

    def gaussian_dropout2d(self, x, is_training, rate=0.5, axis=None):
        return self.gaussian_dropout(x, is_training, rate, axis, 2)

    def gaussian_dropout3d(self, x, is_training, rate=0.5, axis=None):
        return self.gaussian_dropout(x, is_training, rate, axis, 3)
