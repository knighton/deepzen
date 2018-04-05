class BaseDropoutAPI(object):
    def _dropout_mask_shape(self, x_shape, keep_spatial_axis=None):
        if keep_spatial_axis is None:
            keep_spatial_axes = []
        elif isinstance(keep_spatial_axis, int):
            keep_spatial_axes = [keep_spatial_axis]
        elif isinstance(keep_spatial_axis, (list, tuple)):
            keep_spatial_axes = keep_spatial_axis
        else:
            assert False
        mask_shape = [1] * len(x_shape)
        mask_shape[0] = x_shape[0]
        for axis in keep_spatial_axes:
            mask_shape[1 + axis] = x_shape[1 + axis]
        return tuple(mask_shape)

    def dropout(self, x, is_training, rate=0.5, keep_spatial_axis=None,
                xsnd=None):
        if not is_training:
            return x
        if xsnd is not None:
            assert self.spatial_ndim(x) == xsnd
        mask_shape = self._dropout_mask_shape(self.shape(x), keep_spatial_axis)
        mask = self.random_binomial(
            mask_shape, rate, self.dtype(x), self.device(x))
        mask = self.constant(mask)
        return x * mask / (1 - rate)

    def dropout0d(self, x, is_training, rate=0.5, keep_spatial_axis=None):
        return self.dropout(x, is_training, rate, keep_spatial_axis, 0)

    def dropout1d(self, x, is_training, rate=0.5, keep_spatial_axis=None):
        return self.dropout(x, is_training, rate, keep_spatial_axis, 1)

    def dropout2d(self, x, is_training, rate=0.5, keep_spatial_axis=None):
        return self.dropout(x, is_training, rate, keep_spatial_axis, 2)

    def dropout3d(self, x, is_training, rate=0.5, keep_spatial_axis=None):
        return self.dropout(x, is_training, rate, keep_spatial_axis, 3)
