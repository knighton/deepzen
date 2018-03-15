import torch

from ...base.core.logic import BaseLogicAPI


class PyTorchLogicAPI(BaseLogicAPI):
    def minimum(self, a, b):
        return torch.min(a, b)

    def maximum(self, a, b):
        return torch.max(a, b)

    def _fix_compare_dtype(self, a, x):
        return self.cast(x, self.dtype(a))

    def less(self, a, b):
        return self._fix_compare_dtype(a, a < b)

    def less_equal(self, a, b):
        return self._fix_compare_dtype(a, a <= b)

    def equal(self, a, b):
        return self._fix_compare_dtype(a, a == b)

    def greater_equal(self, a, b):
        return self._fix_compare_dtype(a, a >= b)

    def greater(self, a, b):
        return self._fix_compare_dtype(a, a > b)

    def not_equal(self, a, b):
        return self._fix_compare_dtype(a, a != b)
