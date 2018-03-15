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

    def assign_sub(self, x, decr):
        x.data -= decr
        if x.grad is not None:
            x.grad[:] = 0

    def data(self, x):
        return x[:]

    def grad(self, x):
        return x.grad
