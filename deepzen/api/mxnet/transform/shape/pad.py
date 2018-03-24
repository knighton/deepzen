import mxnet as mx

from .....util.unpack import unpack_padding
from ....base.transform.shape.pad import \
    BaseConstantPadAPI, BaseEdgePadAPI, BaseReflectPadAPI, BasePadAPI


class MXNetConstantPadAPI(BaseConstantPadAPI):
    def constant_pad(self, x, padding, value, space=None):
        if space is None:
            space = self.ndim(x) - 2
        if space == 1:
            func = self.constant_pad1d
        elif space == 2:
            func = self.constant_pad2d
        elif space == 3:
            func = self.constant_pad3d
        else:
            assert False
        return func(x, padding, value)

    def constant_pad1d(self, x, padding, value):
        x = mx.nd.expand_dims(x, 2)
        (left, right), = unpack_padding(padding, 1)
        padding = (0, 0), (left, right)
        x = self.constant_pad2d(x, padding, value)
        return mx.nd.squeeze(x, 2)

    def constant_pad2d(self, x, padding, value):
        (top, bottom), (left, right) = unpack_padding(padding, 2)
        mx_padding = 0, 0, 0, 0, top, bottom, left, right
        return mx.nd.pad(x, 'constant', mx_padding, value)

    def constant_pad3d(self, x, padding, value):
        (front, back), (top, bottom), (left, right) = unpack_padding(padding, 3)
        mx_padding = 0, 0, 0, 0, front, back, top, bottom, left, right
        return mx.nd.pad(x, 'constant', mx_padding, value)


class MXNetEdgePadAPI(BaseEdgePadAPI):
    def edge_pad(self, x, padding, space=None):
        if space is None:
            space = self.ndim(x) - 2
        if space == 1:
            func = self.edge_pad1d
        elif space == 2:
            func = self.edge_pad2d
        elif space == 3:
            func = self.edge_pad3d
        else:
            assert False
        return func(x, padding)

    def edge_pad1d(self, x, padding):
        x = mx.nd.expand_dims(x, 2)
        (left, right), = unpack_padding(padding, 1)
        padding = (0, 0), (left, right)
        x = self.edge_pad2d(x, padding)
        return mx.nd.squeeze(x, 2)

    def edge_pad2d(self, x, padding):
        (top, bottom), (left, right) = unpack_padding(padding, 2)
        mx_padding = 0, 0, 0, 0, top, bottom, left, right
        return mx.nd.pad(x, 'edge', mx_padding)

    def edge_pad3d(self, x, padding):
        (front, back), (top, bottom), (left, right) = unpack_padding(padding, 3)
        mx_padding = 0, 0, 0, 0, front, back, top, bottom, left, right
        return mx.nd.pad(x, 'edge', mx_padding)


class MXNetReflectPadAPI(BaseReflectPadAPI):
    def reflect_pad(self, x, padding, space=None):
        if space is None:
            space = self.ndim(x) - 2
        if space == 1:
            func = self.reflect_pad1d
        elif space == 2:
            func = self.reflect_pad2d
        elif space == 3:
            func = self.reflect_pad3d
        else:
            assert False
        return func(x, padding)

    def reflect_pad1d(self, x, padding):
        x = mx.nd.expand_dims(x, 2)
        (left, right), = unpack_padding(padding, 1)
        padding = (0, 0), (left, right)
        x = self.reflect_pad2d(x, padding)
        return mx.nd.squeeze(x, 2)

    def reflect_pad2d(self, x, padding):
        (top, bottom), (left, right) = unpack_padding(padding, 2)
        mx_padding = 0, 0, 0, 0, top, bottom, left, right
        return mx.nd.pad(x, 'reflect', mx_padding)

    def reflect_pad3d(self, x, padding):
        (front, back), (top, bottom), (left, right) = unpack_padding(padding, 3)
        mx_padding = 0, 0, 0, 0, front, back, top, bottom, left, right
        return mx.nd.pad(x, 'reflect', mx_padding)


class MXNetPadAPI(BasePadAPI, MXNetConstantPadAPI, MXNetEdgePadAPI,
                  MXNetReflectPadAPI):
    pass
