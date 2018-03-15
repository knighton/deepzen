from contextlib import contextmanager
import numpy as np
import torch
from torch.autograd import Variable

from . import PyTorchCoreAPI as InnerAPI


class PyTorchCoreAPI(object):
    FLOAT32 = torch.FloatTensor

    def __init__(self, floatx='float32', device=None, epsilon=1e-56):
        self.inner = InnerAPI(floatx, device, epsilon)

    def shape(self, x):
        return tuple(x.size())

    def dtype_of(self, x):
        return self.inner.dtype(x)

    def cast(self, x, dtype):
        return self.inner.cast(x, dtype)

    def flatten(self, x):
        return x.view(x.size()[0], -1)

    def numpy_to_tensor(self, x):
        return self.inner.numpy_to_tensor(x, 'float32')

    def numpy_to_constant(self, x):
        x = self.numpy_to_tensor(x)
        return self.tensor_to_constant(x)

    def numpy_to_variable(self, x):
        x = self.numpy_to_tensor(x)
        return self.tensor_to_variable(x)

    def tensor_to_numpy(self, x):
        if x.is_cuda:
            return x.cpu().numpy()
        else:
            return x.numpy()

    def tensor_to_constant(self, x):
        assert isinstance(x, self.FLOAT32)
        return Variable(x, requires_grad=False)

    def tensor_to_variable(self, x):
        assert isinstance(x, self.FLOAT32)
        return Variable(x, requires_grad=True)

    def zeros(self, shape, dtype):
        assert dtype == 'float32'
        return torch.zeros(shape).type(self.FLOAT32)

    def zeros_like(self, x):
        if isinstance(x, Variable):
            x = x.data
        assert isinstance(x, self.FLOAT32)
        if x.is_cuda:
            return torch.zeros_like(x).cuda()
        else:
            return torch.zeros_like(x)

    def ones(self, shape, dtype):
        assert dtype == 'float32'
        return torch.ones(shape).type(self.FLOAT32)

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

    def backward(self, loss_variables, grad_tensors):
        torch.autograd.backward(loss_variables, grad_tensors)

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
