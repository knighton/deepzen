import mxnet as mx

from .....util.unpack import unpack_dim, unpack_shape
from ....base.layer.shape.upsample import \
    BaseLinearUpsampleAPI, BaseNearestUpsampleAPI, BaseUpsampleAPI


class MXNetLinearUpsampleAPI(BaseLinearUpsampleAPI):
    def linear_upsample(self, x, scale, xsnd=None):
        if xsnd is None:
            xsnd = self.ndim(x) - 2
        if xsnd == 1:
            func = self.linear_upsample1d
        elif xsnd == 2:
            func = self.linear_upsample2d
        elif xsnd == 3:
            func = self.linear_upsample3d
        else:
            assert False
        return func(x, scale)

    def linear_upsample1d(self, x, scale):
        scale = unpack_dim(scale, 1)
        scale = scale, scale
        x = x.unsqueeze(3)
        x = self.linear_upsample2d(x, scale)
        return x[:, :, :, 0]

    def linear_upsample2d(self, x, scale):
        scale = unpack_shape(scale, 2)
        if 1 < len(set(scale)):
            raise 'bummer dude'
        return mx.nd.UpSampling(x, scale=scale[0], sample_type='bilinear')

    def linear_upsample3d(self, x, scale):
        raise 'bummer dude'


class MXNetNearestUpsampleAPI(BaseNearestUpsampleAPI):
    pass


class MXNetUpsampleAPI(BaseUpsampleAPI, MXNetLinearUpsampleAPI,
                       MXNetNearestUpsampleAPI):
    pass
