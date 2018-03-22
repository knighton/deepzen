from contextlib import contextmanager


class BaseVariableAPI(object):
    @contextmanager
    def autograd_record(self):
        raise NotImplementedError

    def backward(self, loss_variables, grad_tensors):
        raise NotImplementedError

    def assign_set(self, x, new_x):
        raise NotImplementedError

    def assign_momentum(self, mov_avg, instance, momentum):
        new_mov_avg = momentum * mov_avg + (1 - momentum) * instance
        self.assign_set(mov_avg, new_mov_avg)

    def assign_sub(self, x, decr):
        raise NotImplementedError

    def ndim(self, x):
        raise NotImplementedError

    def shape(self, x):
        raise NotImplementedError

    def size(self, x):
        raise NotImplementedError

    def data(self, x):
        raise NotImplementedError

    def grad(self, x):
        raise NotImplementedError
