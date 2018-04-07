from torch.nn import functional as F

from ....base.layer.activ.relative import BaseRelativeAPI


class PyTorchRelativeAPI(BaseRelativeAPI):
    def log_softmax(self, x):
        return F.log_softmax(x)

    def softmax(self, x):
        return F.softmax(x, -1)

    def softmin(self, x):
        return F.softmin(x, -1)
