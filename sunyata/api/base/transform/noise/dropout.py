class BaseDropoutAPI(object):
    def _dropout_mask_shape(self, shape, axis=None):
        if axis is None:
            return shape
        if isinstance(axis, int):
            keep_axes = [axis]
        elif isinstance(axis, (list, tuple)):
            keep_axes = axis
        else:
            assert False
        x = [1] * len(shape)
        x[0] = shape[0]
        for axis in keep_axes:
            x[axis] = shape[axis]
        return tuple(x)

    def dropout(self, x, is_training, rate=0.5, axis=None, spatial_ndim=None):
        if not is_training:
            return x
        if spatial_ndim is not None:
            assert self.spatial_ndim(x) == spatial_ndim
        mask_shape = self._dropout_mask_shape(self.shape(x), axis)
        mask = self.random_binomial(
            mask_shape, rate, self.dtype(x), self.device(x))
        mask = self.constant(mask)
        return x * mask / (1 - rate)

    def dropout0d(self, x, is_training, rate=0.5, axis=None):
        return self._dropout(x, is_training, rate, axis, 0)

    def dropout1d(self, x, is_training, rate=0.5, axis=None):
        return self._dropout(x, is_training, rate, axis, 1)

    def dropout2d(self, x, is_training, rate=0.5, axis=None):
        return self._dropout(x, is_training, rate, axis, 2)

    def dropout3d(self, x, is_training, rate=0.5, axis=None):
        return self._dropout(x, is_training, rate, axis, 3)
