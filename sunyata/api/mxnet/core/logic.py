import mxnet as mx

from ...base.core.logic import BaseLogicAPI


class MXNetLogicAPI(BaseLogicAPI):
    def minimum(self, a, b):
        return mx.nd.minimum(a, b)

    def maximum(self, a, b):
        return mx.nd.maximum(a, b)

    def equal(self, a, b):
        return a == b

    def not_equal(self, a, b):
        return a != b

    def less(self, a, b):
        return a < b

    def less_equal(self, a, b):
        return a <= b

    def greater_equal(self, a, b):
        return a >= b

    def greater(self, a, b):
        return a > b
