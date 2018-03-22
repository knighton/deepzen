import mxnet as mx

from ...base.core.tensor import BaseTensorAPI


class MXNetTensorAPI(BaseTensorAPI):
    def zeros(self, shape, dtype=None, device=None):
        dtype = self.dtype(dtype)
        ctx = self.device(device).mx_ctx
        return mx.nd.zeros(shape, ctx, dtype)

    def zeros_like(self, like, dtype=None, device=None):
        dtype = like.dtype if dtype is None else self.dtype(dtype)
        ctx = self.device(device).mx_ctx
        return mx.nd.zeros(like.shape, ctx, dtype)

    def ones(self, shape, dtype=None, device=None):
        dtype = self.dtype(dtype)
        ctx = self.device(device).mx_ctx
        return mx.nd.ones(shape, ctx, dtype)

    def ones_like(self, like, dtype=None, device=None):
        dtype = like.dtype if dtype is None else self.dtype(dtype)
        ctx = self.device(device).mx_ctx
        return mx.nd.ones(like.shape, ctx, dtype)

    def full(self, shape, value, dtype=None, device=None):
        dtype = self.dtype(dtype)
        ctx = self.device(device).mx_ctx
        return mx.nd.full(shape, value, ctx, dtype)

    def full_like(self, like, value, dtype=None, device=None):
        dtype = like.dtype if dtype is None else self.dtype(dtype)
        ctx = self.device(device).mx_ctx
        return mx.nd.full(like.shape, value, ctx, dtype)

    def arange(self, begin, end, step=1, dtype=None, device=None):
        dtype = self.dtype(dtype)
        ctx = self.device(device).mx_ctx
        return mx.nd.arange(begin, end, step, ctx=ctx, dtype=dtype)

    def random_uniform(self, shape, min=0, max=1, dtype=None, device=None):
        dtype = self.dtype(dtype)
        ctx = self.device(device).mx_ctx
        return mx.random.uniform(min, max, shape, ctx=ctx, dtype=dtype)

    def random_normal(self, shape, mean=0, std=1, dtype=None, device=None):
        dtype = self.dtype(dtype)
        ctx = self.device(device).mx_ctx
        return mx.random.normal(mean, std, shape, ctx=ctx, dtype=dtype)

    def random_binomial(self, shape, prob=0.5, dtype=None, device=None):
        dtype = self.dtype(dtype)
        ctx = self.device(device).mx_ctx
        return mx.random.uniform(0, 1, shape, ctx=ctx, dtype=dtype) <= prob
