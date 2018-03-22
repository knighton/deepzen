import mxnet as mx

from ....base.transform.dot.dense import BaseDenseAPI


class MXNetDenseAPI(BaseDenseAPI):
    def dense(self, x, kernel, bias):
        out_dim = kernel.shape[0]
        no_bias = bias is None
        return mx.nd.FullyConnected(x, kernel, bias, out_dim, no_bias)
