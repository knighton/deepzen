from contextlib import contextmanager
from mxnet.autograd import backward, record

from ...base.core.variable import BaseVariableAPI


class MXNetVariableAPI(BaseVariableAPI):
    @contextmanager
    def autograd_record(self):
        with record():
            yield

    def backward(self, loss_variables, grad_tensors):
        backward(loss_variables, grad_tensors)

    def assign_set(self, x, new_x):
        x[:] = x
        if x.grad is not None:
            x.grad[:] = 0

    def assign_sub(self, x, decr):
        x[:] -= decr
        if x.grad is not None:
            x.grad[:] = 0

    def ndim(self, x):
        return x.ndim

    def shape(self, x):
        return x.shape

    def size(self, x):
        return x.size

    def data(self, x):
        return x[:]

    def grad(self, x):
        return x.grad
