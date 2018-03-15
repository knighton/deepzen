import mxnet as mx


class MXNetTransformAPI(object):
    def softmax(self, x):
        return mx.nd.softmax(x)
