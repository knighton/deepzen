from torch.nn import functional as F

from ....base.transform.activ.softmax import BaseSoftmaxAPI


class PyTorchSoftmaxAPI(BaseSoftmaxAPI):
    def softmax(self, x):
        return F.softmax(x, -1)
