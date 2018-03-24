class BaseGlobalAvgPoolAPI(object):
    def global_avg_pool(self, x, space=None):
        return self.global_pool(x, self.mean, space)

    def global_avg_pool1d(self, x):
        return self.global_pool(x, self.mean, 1)

    def global_avg_pool2d(self, x):
        return self.global_pool(x, self.mean, 2)

    def global_avg_pool3d(self, x):
        return self.global_pool(x, self.mean, 3)


class BaseGlobalMaxPoolAPI(object):
    def global_max_pool(self, x, space=None):
        return self.global_pool(x, self.max, space)

    def global_max_pool1d(self, x):
        return self.global_pool(x, self.max, 1)

    def global_max_pool2d(self, x):
        return self.global_pool(x, self.max, 2)

    def global_max_pool3d(self, x):
        return self.global_pool(x, self.max, 3)


class BaseGlobalMinPoolAPI(object):
    def global_min_pool(self, x, space=None):
        return self.global_pool(x, self.min, space)

    def global_min_pool1d(self, x):
        return self.global_pool(x, self.min, 1)

    def global_min_pool2d(self, x):
        return self.global_pool(x, self.min, 2)

    def global_min_pool3d(self, x):
        return self.global_pool(x, self.min, 3)


class BaseGlobalPoolAPI(object):
    def global_pool(self, x, func, space=None):
        if space is not None:
            assert self.spatial_ndim(x) == space
        return func(x, self.spatial_axes(x))

    def global_pool1d(self, x, func):
        return self.global_pool(x, func, 1)

    def global_pool2d(self, x, func):
        return self.global_pool(x, func, 2)

    def global_pool3d(self, x, func):
        return self.global_pool(x, func, 3)

    def global_pool_signature(self, x_sig):
        assert x_sig.has_channels()
        y_sample_shape = x_sig.channels(),
        return x_sig.as_shape(y_sample_shape)
