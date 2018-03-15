from contextlib import contextmanager
from torch.autograd import backward

from ...base.core.variable import BaseVariableAPI


class PyTorchVariableAPI(BaseVariableAPI):
    @contextmanager
    def autograd_record(self):
        yield

    def backward(self, loss_variables, grad_tensors):
        backward(loss_variables, grad_tensors)

    def assign_sub(self, x, decr):
        x.data -= decr
        x.grad.data.zero_()

    def data(self, x):
        return x.data

    def grad(self, x):
        return x.grad.data
