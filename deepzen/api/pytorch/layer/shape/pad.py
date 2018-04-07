from torch.nn import functional as F

from .....util.unpack import unpack_padding
from ....base.layer.shape.pad import \
    BaseConstantPadAPI, BaseEdgePadAPI, BaseReflectPadAPI, BasePadAPI


class PyTorchConstantPadAPI(BaseConstantPadAPI):
    def constant_pad(self, x, padding, value, xsnd=None):
        if xsnd is None:
            xsnd = self.ndim(x) - 2
        if xsnd == 1:
            func = self.constant_pad1d
        elif xsnd == 2:
            func = self.constant_pad2d
        elif xsnd == 3:
            func = self.constant_pad3d
        else:
            assert False
        return func(x, padding, value)

    def constant_pad1d(self, x, padding, value):
        x = x.unsqueeze(2)
        (left, right), = unpack_padding(padding, 1)
        pt_padding = 0, 0, left, right
        x = F.pad(x, pt_padding, 'constant', value)
        return x.squeeze(2)

    def constant_pad2d(self, x, padding, value):
        (top, bottom), (left, right) = unpack_padding(padding, 2)
        pt_padding = top, bottom, left, right
        return F.pad(x, pt_padding, 'constant', value)

    def constant_pad3d(self, x, padding, value):
        (front, back), (top, bottom), (left, right) = unpack_padding(padding, 3)
        pt_padding = front, back, top, bottom, left, right
        return F.pad(x, pt_padding, 'constant', value)


class PyTorchEdgePadAPI(BaseEdgePadAPI):
    def edge_pad(self, x, padding, xsnd=None):
        if xsnd is None:
            xsnd = self.ndim(x) - 2
        if xsnd == 1:
            func = self.edge_pad1d
        elif xsnd == 2:
            func = self.edge_pad2d
        elif xsnd == 3:
            func = self.edge_pad3d
        else:
            assert False
        return func(x, padding)

    def edge_pad1d(self, x, padding):
        x = x.unsqueeze(2)
        (left, right), = unpack_padding(padding, 1)
        pt_padding = 0, 0, left, right
        x = F.pad(x, pt_padding, 'replicate')
        return x.squeeze(2)

    def edge_pad2d(self, x, padding):
        (top, bottom), (left, right) = unpack_padding(padding, 2)
        pt_padding = top, bottom, left, right
        return F.pad(x, pt_padding, 'replicate')

    def edge_pad3d(self, x, padding):
        (front, back), (top, bottom), (left, right) = unpack_padding(padding, 3)
        pt_padding = front, back, top, bottom, left, right
        return F.pad(x, pt_padding, 'replicate')


class PyTorchReflectPadAPI(BaseReflectPadAPI):
    def reflect_pad(self, x, padding, xsnd=None):
        if xsnd is None:
            xsnd = self.ndim(x) - 2
        if xsnd == 1:
            func = self.reflect_pad1d
        elif xsnd == 2:
            func = self.reflect_pad2d
        elif xsnd == 3:
            func = self.reflect_pad3d
        else:
            assert False
        return func(x, padding)

    def reflect_pad1d(self, x, padding):
        x = x.unsqueeze(2)
        (left, right), = unpack_padding(padding, 1)
        pt_padding = 0, 0, left, right
        x = F.pad(x, pt_padding, 'reflect')
        return x.squeeze(2)

    def reflect_pad2d(self, x, padding):
        (top, bottom), (left, right) = unpack_padding(padding, 2)
        pt_padding = top, bottom, left, right
        return F.pad(x, pt_padding, 'reflect')

    def reflect_pad3d(self, x, padding):
        (front, back), (top, bottom), (left, right) = unpack_padding(padding, 3)
        pt_padding = front, back, top, bottom, left, right
        return F.pad(x, pt_padding, 'reflect')


class PyTorchPadAPI(BasePadAPI, PyTorchConstantPadAPI, PyTorchEdgePadAPI,
                    PyTorchReflectPadAPI):
    pass
