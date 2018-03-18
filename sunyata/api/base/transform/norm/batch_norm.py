class BaseBatchNormAPI(object):
    def batch_norm(self, x, beta, gamma, mean, var):
        x = (x - mean) / self.sqrt(var + self.epsilon())
        if gamma is not None:
            x *= gamma
        if beta is not None:
            x += beta
        return x

    @classmethod
    def _batch_norm_reduction_axes(self, batch_shape):
        axes = []
        for i, axis in enumerate(batch_shape):
            if axis == 1:
                axes.append(axis)
        return axes

    def instance_batch_norm(self, x, beta, gamma, space=None):
        if space is not None:
            assert self.spatial_ndim(x) == space
        reduction_axes = self._batch_norm_reduction_axes(self.shape(beta))
        mean, var = self.moments(x, reduction_axes)
        return self.batch_norm(x, beta, gamma, mean, var)

    def instance_batch_norm0d(self, x, beta, gamma):
        return self.instance_batch_norm(x, beta, gamma, 0)

    def instance_batch_norm1d(self, x, beta, gamma):
        return self.instance_batch_norm(x, beta, gamma, 1)

    def instance_batch_norm2d(self, x, beta, gamma):
        return self.instance_batch_norm(x, beta, gamma, 2)

    def instance_batch_norm3d(self, x, beta, gamma):
        return self.instance_batch_norm(x, beta, gamma, 3)

    def mov_avg_batch_norm(self, x, is_training, momentum, beta, gamma,
                           mov_avg_mean, mov_avg_var, space=None):
        if space is not None:
            assert self.spatial_ndim(x) == space
        if not is_training:
            return self.batch_norm(x, beta, gamma, mov_avg_mean, mov_avg_var)
        reduction_axes = self._batch_norm_reduction_axes(self.shape(beta))
        instance_mean, instance_var = self.moments(x, reduction_axes)
        x = self.batch_norm(x, beta, gamma, instance_mean, instance_var)
        self.assign_momentum(mov_avg_mean, instance_mean, momentum)
        self.assign_momentum(mov_avg_var, instance_var, momentum)
        return self.batch_norm(x, beta, gamma, mov_avg_mean, mov_avg_var)

    def mov_avg_batch_norm0d(self, x, is_training, momentum, beta, gamma,
                             mov_avg_mean, mov_avg_var):
        return self.mov_avg_batch_norm(x, is_training, momentum, beta, gamma,
                                       mov_avg_mean, mov_avg_var, 0)

    def mov_avg_batch_norm1d(self, x, is_training, momentum, beta, gamma,
                             mov_avg_mean, mov_avg_var):
        return self.mov_avg_batch_norm(x, is_training, momentum, beta, gamma,
                                       mov_avg_mean, mov_avg_var, 1)

    def mov_avg_batch_norm2d(self, x, is_training, momentum, beta, gamma,
                             mov_avg_mean, mov_avg_var):
        return self.mov_avg_batch_norm(x, is_training, momentum, beta, gamma,
                                       mov_avg_mean, mov_avg_var, 2)

    def mov_avg_batch_norm3d(self, x, is_training, momentum, beta, gamma,
                             mov_avg_mean, mov_avg_var):
        return self.mov_avg_batch_norm(x, is_training, momentum, beta, gamma,
                                       mov_avg_mean, mov_avg_var, 3)
