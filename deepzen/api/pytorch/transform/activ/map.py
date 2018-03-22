from torch.nn import functional as F

from ....base.transform.activ.map import BaseMapAPI


class PyTorchMapAPI(BaseMapAPI):
    def elu(self, x, alpha=1):
        return F.elu(x, alpha)

    def leaky_relu(self, x, alpha=0.1):
        return F.leaky_relu(x, alpha)

    def log_sigmoid(self, x):
        return F.logsigmoid(x)

    def log_softmax(self, x):
        return F.log_softmax(x)

    def selu(self, x):
        return F.selu(x)

    def sigmoid(self, x):
        return F.sigmoid(x)

    def softplus(self, x, beta=1, threshold=20):
        return F.softplus(x, beta, threshold)

    def softshrink(self, x, lambd=0.5):
        return F.softshrink(x, lambd)

    def softsign(self, x):
        return F.softsign(x)

    def tanh_shrink(self, x):
        return F.tanhshrink(x)
