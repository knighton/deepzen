import mxnet as mx

from ....base.transform.activ.softmax import BaseSoftmaxAPI


class MXNetSoftmaxAPI(BaseSoftmaxAPI):
    def softmax(self, x):
        return mx.nd.softmax(x)
