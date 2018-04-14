from torch.nn import functional as F

from .....util.unpack import unpack_dim
from ....base.layer.dot.conv import BaseConvAPI


class PyTorchConvAPI(BaseConvAPI):
    def conv(self, x, kernel, bias, stride, padding, dilation, xsnd=None):
        if xsnd is None:
            xsnd = x.dim() - 2
        if xsnd == 1:
            func = self.conv1d
        elif xsnd == 2:
            func = self.conv2d
        elif xsnd == 3:
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

    def conv_transpose(self, x, kernel, bias, stride, padding, out_padding,
                       dilation, xsnd=None):
        if xsnd is None:
            xsnd = x.dim() - 2
        if xsnd == 1:
            func = self.conv_transpose1d
        elif xsnd == 2:
            func = self.conv_transpose2d
        elif xsnd == 3:
            func = self.conv_transpose3d
        else:
            assert False
        return func(x, kernel, bias, stride, padding, out_padding, dilation)

    def conv_transpose1d(self, x, kernel, bias, stride, padding, out_padding,
                         dilation):
        stride = unpack_dim(stride)
        padding = unpack_dim(padding)
        out_padding = unpack_dim(out_padding)
        groups = 1
        dilation = unpack_dim(dilation)
        return F.conv_transpose1d(x, kernel, bias, stride, padding, out_padding,
                                  groups, dilation)

    def conv_transpose2d(self, x, kernel, bias, stride, padding, out_padding,
                         dilation):
        groups = 1
        return F.conv_transpose2d(x, kernel, bias, stride, padding, out_padding,
                                  groups, dilation)

    def conv_transpose3d(self, x, kernel, bias, stride, padding, out_padding,
                         dilation):
        groups = 1
        return F.conv_transpose3d(x, kernel, bias, stride, padding, out_padding,
                                  groups, dilation)
