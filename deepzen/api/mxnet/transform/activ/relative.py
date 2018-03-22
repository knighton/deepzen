import mxnet as mx

from ....base.transform.activ.relative import BaseRelativeAPI


class MXNetRelativeAPI(BaseRelativeAPI):
    def softmax(self, x):
        return mx.nd.softmax(x)
