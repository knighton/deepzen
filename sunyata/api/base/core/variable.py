from contextlib import contextmanager


class BaseVariableAPI(object):
    @contextmanager
    def autograd_record(self):
        raise NotImplementedError

    def backward(self, loss_variables, grad_tensors):
        raise NotImplementedError

    def assign_sub(self, x, decr):
        raise NotImplementedError

    def data(self, x):
        raise NotImplementedError

    def grad(self, x):
        raise NotImplementedError
