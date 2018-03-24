from .....util.unpack import unpack_padding


class BaseConstantPadAPI(object):
    def constant_pad(self, x, padding, value, space=None):
        raise NotImplementedError

    def constant_pad1d(self, x, padding, value):
        raise NotImplementedError

    def constant_pad2d(self, x, padding, value):
        raise NotImplementedError

    def constant_pad3d(self, x, padding, value):
        raise NotImplementedError


class BaseEdgePadAPI(object):
    def edge_pad(self, x, padding, space=None):
        raise NotImplementedError

    def edge_pad1d(self, x, padding):
        raise NotImplementedError

    def edge_pad2d(self, x, padding):
        raise NotImplementedError

    def edge_pad3d(self, x, padding):
        raise NotImplementedError


class BaseReflectPadAPI(object):
    def reflect_pad(self, x, padding, space=None):
        raise NotImplementedError

    def reflect_pad1d(self, x, padding):
        raise NotImplementedError

    def reflect_pad2d(self, x, padding):
        raise NotImplementedError

    def reflect_pad3d(self, x, padding):
        raise NotImplementedError


class BasePadAPI(BaseConstantPadAPI, BaseEdgePadAPI, BaseReflectPadAPI):
    def pad_signature(self, x_sig, padding):
        assert x_sig.has_channels()
        padding = unpack_padding(padding, x_sig.spatial_ndim())
        y_sample_shape = [x_sig.channels()]
        for dim, (left, right) in zip(x_sig.spatial_shape(), padding):
            y_sample_shape.append(left + dim + right)
        return x_sig.as_shape(y_sample_shape)
