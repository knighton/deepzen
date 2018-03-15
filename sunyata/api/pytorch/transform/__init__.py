from torch.nn import functional as F

from ...base import API


class PyTorchTransformAPI(API):
    def softmax(self, x):
        return F.softmax(x, -1)
