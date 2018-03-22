import mxnet as mx

from ....base.transform.activ.map import BaseMapAPI


class MXNetMapAPI(BaseMapAPI):
    def elu(self, x, alpha=1.):
        return mx.nd.LeakyReLU(x, 'elu', alpha)

    def leaky_relu(self, x, alpha=0.1):
        return mx.nd.LeakyReLU(x, 'leaky', alpha)

    def sigmoid(self, x):
        return mx.nd.sigmoid(x)

    def softplus(self, x):
        return mx.nd.Activation(x, 'softrelu')
