from ...base.core.reduce import BaseReduceAPI


class PyTorchReduceAPI(BaseReduceAPI):
    def argmin(self, x, axis=-1):
        return x.min(axis)[1]

    def argmax(self, x, axis=-1):
        return x.max(axis)[1]

    def _normalize_reduce_axis(self, axis, keepdims, ndim):
        if axis is None:
            axes = list(range(ndim))
        elif isinstance(axis, int):
            axes = [axis % ndim]
        elif isinstance(axis, (list, tuple)):
            axes = sorted(list(map(lambda n: n % ndim, axis)))
        else:
            assert False
        return axes

    def _reduce(self, name, x, axis=None, keepdims=False):
        axes = self._normalize_reduce_axis(axis, keepdims, x.dim())
        for axis in reversed(axes):
            if x.dim() == 1:
                keepdims = True
            x = getattr(x, name)(axis, keepdims)
            if isinstance(x, tuple):
                x = x[0]
        return x

    def min(self, x, axis=None, keepdims=False):
        return self._reduce('min', x, axis, keepdims)

    def max(self, x, axis=None, keepdims=False):
        return self._reduce('max', x, axis, keepdims)

    def mean(self, x, axis=None, keepdims=False):
        return self._reduce('mean', x, axis, keepdims)

    def sum(self, x, axis=None, keepdims=False):
        return self._reduce('sum', x, axis, keepdims)

    def prod(self, x, axis=None, keepdims=False):
        return self._reduce('prod', x, axis, keepdims)
