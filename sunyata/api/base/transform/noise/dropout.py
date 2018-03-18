class BaseDropoutAPI(object):
    def _dropout_mask_shape(self, x_shape, axis=None):
        if axis is None:
            return x_shape
        if isinstance(axis, int):
            axes = [axis]
        elif isinstance(axis, (list, tuple)):
            axes = axis
        else:
            assert False
        mask_shape = [1] * len(x_shape)
        mask_shape[0] = x_shape[0]
        for axis in axes:
            mask_shape[1 + axis] = x_shape[1 + axis]
        return tuple(mask_shape)

    def dropout(self, x, is_training, rate=0.5, axis=None, space=None):
        if not is_training:
            return x
        if space is not None:
            assert self.spatial_ndim(x) == space
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
