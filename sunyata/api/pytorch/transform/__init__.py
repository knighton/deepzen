from torch.nn import functional as F


class PyTorchTransformAPI(object):
    def softmax(self, x):
        return F.softmax(x, -1)
