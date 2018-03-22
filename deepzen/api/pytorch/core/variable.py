from contextlib import contextmanager
from torch.autograd import backward

from ...base.core.variable import BaseVariableAPI


class PyTorchVariableAPI(BaseVariableAPI):
    @contextmanager
    def autograd_record(self):
        yield

    def backward(self, loss_variables, grad_tensors):
        backward(loss_variables, grad_tensors)

    def assign_set(self, x, new_x):
        x = new_x
        if x.grad is not None:
            x.grad.data.zero_()

    def assign_sub(self, x, decr):
        x.data -= decr
        if x.grad is not None:
            x.grad.data.zero_()

    def ndim(self, x):
        return x.dim()

    def shape(self, x):
        return tuple(x.size())

    def size(self, x):
        return x.numel()

    def data(self, x):
        return x.data

    def grad(self, x):
        return x.grad.data
