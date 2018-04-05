class BaseDropoutAPI(object):
    def _dropout_mask_shape(self, x_shape, keep_saxis=None):
        if keep_saxis is None:
            keep_saxes = []
        elif isinstance(keep_saxis, int):
            keep_saxes = [keep_saxis]
        elif isinstance(keep_saxis, (list, tuple)):
            keep_saxes = keep_saxis
        else:
            assert False
        mask_shape = [1] * len(x_shape)
        mask_shape[0] = x_shape[0]
        for axis in keep_saxes:
            mask_shape[1 + axis] = x_shape[1 + axis]
        return tuple(mask_shape)

    def dropout(self, x, is_training, rate=0.5, keep_saxis=None, xsnd=None):
        if not is_training:
            return x
        if xsnd is not None:
            assert self.spatial_ndim(x) == xsnd
        mask_shape = self._dropout_mask_shape(self.shape(x), keep_saxis)
        mask = self.random_binomial(
            mask_shape, rate, self.dtype(x), self.device(x))
        mask = self.constant(mask)
        return x * mask / (1 - rate)

    def dropout0d(self, x, is_training, rate=0.5, keep_saxis=None):
        return self.dropout(x, is_training, rate, keep_saxis, 0)

    def dropout1d(self, x, is_training, rate=0.5, keep_saxis=None):
        return self.dropout(x, is_training, rate, keep_saxis, 1)

    def dropout2d(self, x, is_training, rate=0.5, keep_saxis=None):
        return self.dropout(x, is_training, rate, keep_saxis, 2)

    def dropout3d(self, x, is_training, rate=0.5, keep_saxis=None):
        return self.dropout(x, is_training, rate, keep_saxis, 3)
