class BaseCastAPI(object):
    def _init_api_base_core_cast(self):
        pass

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

    def numpy_to_tensor(self, x, dtype=None, device=None):
        raise NotImplementedError
