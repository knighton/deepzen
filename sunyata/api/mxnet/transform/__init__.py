import mxnet as mx

from ...base import API


class MXNetTransformAPI(API):
    def softmax(self, x):
        return mx.nd.softmax(x)
