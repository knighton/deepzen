class BaseAlphaDropoutAPI(object):
    _alpha_dropout_alpha = 1.6732632423543772848170429916717
    _alpha_dropout_scale = 1.0507009873554804934193349852946
    _alpha_dropout_alpha_p = -_alpha_dropout_alpha * _alpha_dropout_scale

    def alpha_dropout(self, x, is_training, rate=0.5, keep_spatial_axis=None,
                      xsnd=None):
        # Only drop out in training mode.
        if not is_training:
            return x

        # Apply optional input dimensionality restriction.
        if xsnd is not None:
            assert self.spatial_ndim(x) == xsnd

        # Create the mask.
        mask_shape = self._dropout_mask_shape(self.shape(x), keep_spatial_axis)
        mask = self.random_binomial(mask_shape, rate, self.dtype(x),
                                    self.device(x))
        mask = self.constant(mask)

        # Get affine transformation params.
        left = 1 - rate
        right = 1 + rate * self.square(self._alpha_dropout_alpha_p)
        a = self.rsqrt(left * right)
        b = -a * self._alpha_dropout_alpha_p * rate

        # Apply mask.
        x = mask * x + (1 - mask) * self._alpha_dropout_alpha_p

        # Do affine transformation.
        return a * x + b

    def alpha_dropout0d(self, x, is_training, rate=0.5, keep_spatial_axis=None):
        return self.alpha_dropout(x, is_training, rate, keep_spatial_axis, 0)

    def alpha_dropout1d(self, x, is_training, rate=0.5, keep_spatial_axis=None):
        return self.alpha_dropout(x, is_training, rate, keep_spatial_axis, 1)

    def alpha_dropout2d(self, x, is_training, rate=0.5, keep_spatial_axis=None):
        return self.alpha_dropout(x, is_training, rate, keep_spatial_axis, 2)

    def alpha_dropout3d(self, x, is_training, rate=0.5, keep_spatial_axis=None):
        return self.alpha_dropout(x, is_training, rate, keep_spatial_axis, 3)
