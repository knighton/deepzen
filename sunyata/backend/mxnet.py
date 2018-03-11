from contextlib import contextmanager
import mxnet as mx
import numpy as np

from .backend import Backend


class MXNetBackend(object):
    def shape(self, x):
        return x.shape

    def dtype_of(self, x):
        return x.dtype.__name__

    def cast(self, x, dtype):
        return x.astype(dtype)

    def flatten(self, x):
        return mx.nd.flatten(x)

    def numpy_to_tensor(self, x):
        assert isinstance(x, np.ndarray)
        return mx.nd.array(x)

    def numpy_to_constant(self, x):
        assert isinstance(x, np.ndarray)
        return mx.nd.array(x)

    def numpy_to_variable(self, x):
        assert isinstance(x, np.ndarray)
        x = mx.nd.array(x)
        x.attach_grad()
        return x

    def tensor_to_numpy(self, x):
        assert isinstance(x, mx.nd.NDArray)
        return x.asnumpy()

    def tensor_to_constant(self, x):
        assert isinstance(x, mx.nd.NDArray)
        return x.copy()

    def tensor_to_variable(self, x):
        assert isinstance(x, mx.nd.NDArray)
        x = x.copy()
        x.attach_grad()
        return x

    def zeros(self, shape, dtype):
        assert dtype == 'float32'
        return mx.nd.zeros(shape)

    def zeros_like(self, x):
        return mx.nd.zeros_like(x)

    def ones(self, shape, dtype):
        assert dtype == 'float32'
        return mx.nd.ones(shape)

    def ones_like(self, x):
        return mx.nd.ones_like(x)

    @contextmanager
    def autograd_record(self):
        with mx.autograd.record():
            yield

    def backward(self, loss_variables, grad_tensors):
        mx.autograd.backward(loss_variables, grad_tensors)

    def constant_to_numpy(self, x):
        return x.asnumpy()

    def variable_to_numpy(self, x):
        return x.asnumpy()

    def matmul(self, a, b):
        return mx.nd.dot(a, b)

    def clip(self, x, min=-np.inf, max=np.inf):
        return mx.nd.clip(x, min, max)

    def softmax(self, x):
        return mx.nd.softmax(x)

    def assign_sub(self, x, decr):
        x -= decr
        if x.grad is not None:
            x.grad[:] = 0

    def data(self, x):
        return x[:]

    def grad(self, x):
        return x.grad

    def pow(self, x, power):
        return x ** power

    def sum(self, x):
        return mx.nd.sum(x)

    def log(self, x):
        return mx.nd.log(x)

    def mean(self, x):
        return mx.nd.mean(x)

    def equal(self, a, b):
        return mx.nd.equal(a, b)

    def argmax(self, x, axis=-1):
        return mx.nd.argmax(x, axis)
