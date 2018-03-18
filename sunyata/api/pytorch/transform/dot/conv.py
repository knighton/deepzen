from torch.nn import functional as F

from .....util.unpack import unpack_dim
from ....base.transform.dot.conv import BaseConvAPI


class PyTorchConvAPI(BaseConvAPI):
    def conv(self, x, kernel, bias, stride, padding, dilation, space=None):
        if space is None:
            space = x.dim() - 2
        if space == 1:
            func = self.conv1d
        elif space == 2:
            func = self.conv2d
        elif space == 3:
            func = self.conv3d
        else:
            assert False
        return func(x, kernel, bias, stride, padding, dilation)

    def conv1d(self, x, kernel, bias, stride, padding, dilation):
        stride = unpack_dim(stride)
        padding = unpack_dim(padding)
        dilation = unpack_dim(dilation)
        return F.conv1d(x, kernel, bias, stride, padding, dilation)

    def conv2d(self, x, kernel, bias, stride, padding, dilation):
        return F.conv2d(x, kernel, bias, stride, padding, dilation)

    def conv3d(self, x, kernel, bias, stride, padding, dilation):
        return F.conv3d(x, kernel, bias, stride, padding, dilation)
