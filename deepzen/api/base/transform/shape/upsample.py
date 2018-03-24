from .....util.unpack import unpack_shape


class BaseLinearUpsampleAPI(object):
    def linear_upsample(self, x, scale, space=None):
        raise NotImplementedError

    def linear_upsample1d(self, x, scale):
        raise NotImplementedError

    def linear_upsample2d(self, x, scale):
        raise NotImplementedError

    def linear_upsample3d(self, x, scale):
        raise NotImplementedError


class BaseNearestUpsampleAPI(object):
    def nearest_upsample(self, x, scale, space=None):
        if space is None:
            space = self.ndim(x) - 2
        else:
            assert space == self.ndim(x) - 2
        scale = unpack_shape(scale, space)
        return self.repeat(x, (1, 1) + scale)

    def nearest_upsample1d(self, x, scale):
        return self.nearest_upsample(x, scale, 1)

    def nearest_upsample2d(self, x, scale):
        return self.nearest_upsample(x, scale, 2)

    def nearest_upsample3d(self, x, scale):
        return self.nearest_upsample(x, scale, 3)


class BaseUpsampleAPI(BaseLinearUpsampleAPI, BaseNearestUpsampleAPI):
    def upsample_signaure(self, x_sig, scale):
        assert x_sig.has_channels()
        scale = unpack_shape(scale, x_sig.spatial_ndim())
        y_sample_shape = [x_sig.channels()]
        for dim, mul in zip(x_sig.spatial_shape(), scale):
            y_sample_shape.append(dim * mul)
        return x_sig.as_shape(y_sample_shape)
