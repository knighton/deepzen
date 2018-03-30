class BaseGaussianNoiseAPI(object):
    def gaussian_noise(self, x, is_training, std, axis=None, xsnd=None):
        if not is_training:
            return x
        if xsnd is not None:
            assert self.spatial_ndim(x) == xsnd
        add_shape = self._dropout_mask_shape(self.shape(x), axis)
        add = self.random_normal(add_shape, 0, std)
        return x + self.constant(add)

    def gaussian_noise0d(self, x, is_training, std, axis=None):
        return self.gaussian_noise(x, is_training, std, axis, 0)

    def gaussian_noise1d(self, x, is_training, std, axis=None):
        return self.gaussian_noise(x, is_training, std, axis, 1)

    def gaussian_noise2d(self, x, is_training, std, axis=None):
        return self.gaussian_noise(x, is_training, std, axis, 2)

    def gaussian_noise3d(self, x, is_training, std, axis=None):
        return self.gaussian_noise(x, is_training, std, axis, 3)
