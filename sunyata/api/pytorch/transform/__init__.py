from torch.nn import functional as F

from ...base.transform import BaseTransformAPI


class PyTorchTransformAPI(BaseTransformAPI):
    def softmax(self, x):
        return F.softmax(x, -1)

    def dense(self, x, kernel, bias):
        return F.linear(x, kernel, bias)
