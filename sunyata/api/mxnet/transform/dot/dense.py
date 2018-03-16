import mxnet as mx

from ....base.transform.dot.dense import BaseDenseAPI


class MXNetDenseAPI(BaseDenseAPI):
    def dense(self, x, kernel, bias):
        return mx.nd.FullyConnected(x, kernel, bias, bias.shape[0])
