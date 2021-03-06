from torch.nn import functional as F

from .....util.unpack import unpack_shape
from ....base.layer.shape.upsample import \
    BaseLinearUpsampleAPI, BaseNearestUpsampleAPI, BaseUpsampleAPI


class PyTorchLinearUpsampleAPI(BaseLinearUpsampleAPI):
    def linear_upsample(self, x, scale, xsnd=None):
        if xsnd is None:
            xsnd = self.ndim(x) - 2
        scale = unpack_shape(scale, xsnd)
        if 1 < len(set(scale)):
            raise 'bummer dude'
        method = ['linear', 'bilinear', 'trilinear'][xsnd - 1]
        return F.upsample(x, None, scale[0], method)

    def linear_upsample1d(self, x, scale):
        return self.linear_upsample(x, scale, 1)

    def linear_upsample2d(self, x, scale):
        return self.linear_upsample(x, scale, 2)

    def linear_upsample3d(self, x, scale):
        return self.linear_upsample(x, scale, 3)


class PyTorchNearestUpsampleAPI(BaseNearestUpsampleAPI):
    pass


class PyTorchUpsampleAPI(BaseUpsampleAPI, PyTorchLinearUpsampleAPI,
                         PyTorchNearestUpsampleAPI):
    pass
