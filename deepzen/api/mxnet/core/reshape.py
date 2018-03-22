import mxnet as mx

from ...base.core.reshape import BaseReshapeAPI


class MXNetReshapeAPI(BaseReshapeAPI):
    def reshape(self, x, shape):
        return mx.nd.reshape(x, shape)

    def expand_dims(self, x, axis):
        return mx.nd.expand_dims(x, axis)

    def squeeze(self, x, axis=None):
        from_shape = tuple(x.shape)
        assert -len(from_shape) < axis < len(from_shape)
        axis %= len(from_shape)
        assert from_shape[axis] == 1
        to_shape = from_shape[:axis] + from_shape[axis + 1:]
        return mx.nd.reshape(x, to_shape)

    def repeat_axis(x, axis, repeat):
        return mx.nd.repeat(x, repeat, axis)

    def tile_axis(self, x, axis, repeat):
        repeats = [1] * self.ndim(x)
        repeats[axis] = repeat
        return mx.nd.tile(x, repeats)

    def tile(self, x, repeats):
        return mx.nd.tile(x, repeats)

    def transpose(self, x, axes):
        return mx.nd.transpose(x, axes)

    def split(self, x, axis):
        return mx.nd.split(x, x.shape[axis], axis)

    def concat(self, xx, axis):
        return mx.nd.concat(*xx, dim=axis)

    def stack(self, xx, axis=0):
        return mx.nd.stack(*xx, axis=axis)
