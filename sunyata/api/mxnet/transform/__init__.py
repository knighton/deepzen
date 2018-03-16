import mxnet as mx

from ...base.transform import BaseTransformAPI


class MXNetTransformAPI(BaseTransformAPI):
    def softmax(self, x):
        return mx.nd.softmax(x)

    def dense(self, x, kernel, bias):
        return mx.nd.FullyConnected(x, kernel, bias, bias.shape[0])
