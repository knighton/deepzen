class BaseCastAPI(object):
    def cast(self, x, dtype=None, device=None, copy=False):
        raise NotImplementedError

    def cast_to_cpu(self, x, dtype, copy=False):
        return self.cast(x, dtype, self.cpu(), copy)

    def cast_to_gpu(self, x, dtype, gpu=None, copy=False):
        return self.cast(x, dtype, self.gpu(gpu), copy)

    def to_device(self, x, device=None, copy=False):
        return self.cast(x, self.dtype(x), device, copy)

    def to_cpu(self, x, copy=False):
        return self.cast(x, self.dtype(x), self.cpu(), copy)

    def to_gpu(self, x, gpu=None, copy=False):
        return self.cast(x, self.dtype(x), self.gpu(gpu), copy)

    def cast_numpy_to_tensor(self, x, dtype=None, device=None):
        raise NotImplementedError

    def numpy_to_tensor(self, x, device=None):
        return self.cast_numpy_to_tensor(x, x.dtype.name, device)
