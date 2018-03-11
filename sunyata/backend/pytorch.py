from contextlib import contextmanager
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F

from .backend import Backend


class PyTorchBackend(Backend):
    FLOAT32 = torch.FloatTensor

    def shape(self, x):
        return tuple(x.size())

    def dtype_of(self, x):
        assert isinstance(x.data, self.FLOAT32)
        return 'float32'

    def cast(self, x, dtype):
        assert dtype == 'float32'
        return x.type(self.FLOAT32)

    def flatten(self, x):
        return x.view(x.size()[0], -1)

    def tensor(self, x):
        assert isinstance(x, np.ndarray)
        return torch.from_numpy(x).type(self.FLOAT32)

    def constant(self, x):
        assert isinstance(x, self.FLOAT32)
        return Variable(x, requires_grad=False)

    def variable(self, x):
        assert isinstance(x, self.FLOAT32)
        return Variable(x, requires_grad=True)

    def zeros_like(self, x):
        if isinstance(x, Variable):
            x = x.data
        assert isinstance(x, self.FLOAT32)
        if x.is_cuda:
            return torch.zeros_like(x).cuda()
        else:
            return torch.zeros_like(x)

    def ones_like(self, x):
        if isinstance(x, Variable):
            x = x.data
        assert isinstance(x, self.FLOAT32)
        if x.is_cuda:
            return torch.ones_like(x).cuda()
        else:
            return torch.ones_like(x)

    @contextmanager
    def autograd_record(self):
        yield

    def tensor_to_numpy(self, x):
        if x.is_cuda:
            return x.cpu().numpy()
        else:
            return x.numpy()

    def constant_to_numpy(self, x):
        if x.data.is_cuda:
            return x.data.cpu().numpy()
        else:
            return x.data.numpy()

    def variable_to_numpy(self, x):
        if x.data.is_cuda:
            return x.data.cpu().numpy()
        else:
            return x.data.numpy()

    def matmul(self, a, b):
        return a.mm(b)

    def clip(self, x, min=-np.inf, max=np.inf):
        return x.clamp(min, max)

    def softmax(self, x):
        return F.softmax(x, -1)

    def assign_sub(self, x, decr):
        x.data -= decr
        x.grad.data.zero_()

    def grad(self, x):
        return x.grad.data

    def pow(self, x, power):
        return x.pow(power)

    def sum(self, x):
        return x.sum()

    def log(self, x):
        return x.log()

    def mean(self, x):
        return x.mean()

    def equal(self, a, b):
        return a == b

    def argmax(self, x, axis=-1):
        return x.max(axis)[1]
