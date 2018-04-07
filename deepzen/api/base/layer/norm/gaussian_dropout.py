import numpy as np


class BaseGaussianDropoutAPI(object):
    def gaussian_dropout(self, x, is_training, rate=0.5, keep_spatial_axis=None,
                         xsnd=None):
        if not is_training:
            return x
        if xsnd is not None:
            assert self.spatial_ndim(x) == xsnd
        mul_shape = self._dropout_mask_shape(self.shape(x), keep_spatial_axis)
        std = np.sqrt(rate / (1 - rate))
        mul = self.random_normal(mul_shape, 1, std)
        return x * self.constant(mul)

    def gaussian_dropout0d(self, x, is_training, rate=0.5,
                           keep_spatial_axis=None):
        return self.gaussian_dropout(x, is_training, rate, keep_spatial_axis, 0)

    def gaussian_dropout1d(self, x, is_training, rate=0.5,
                           keep_spatial_axis=None):
        return self.gaussian_dropout(x, is_training, rate, keep_spatial_axis, 1)

    def gaussian_dropout2d(self, x, is_training, rate=0.5,
                           keep_spatial_axis=None):
        return self.gaussian_dropout(x, is_training, rate, keep_spatial_axis, 2)

    def gaussian_dropout3d(self, x, is_training, rate=0.5,
                           keep_spatial_axis=None):
        return self.gaussian_dropout(x, is_training, rate, keep_spatial_axis, 3)
