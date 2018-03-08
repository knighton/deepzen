import numpy as np
import torch
from torch.autograd import Variable


def dtype_of(x):
    assert isinstance(x.data, torch.FloatTensor)
    return 'float32'


class Form(object):
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    def equals(self, other):
        return self.shape == other.shape and self.dtype == other.dtype

    def check(self, x):
        assert tuple(x.size()[1:]) == self.shape
        assert dtype_of(x) == self.dtype


class Layer(object):
    def params(self):
        return []

    def forward(self, x):
        raise NotImplementedError


class DataLayer(Layer):
    def __init__(self, form):
        self.form = form

    def forward(self, x):
        self.form.check(x)
        return x


class DenseLayer(Layer):
    def __init__(self, kernel, bias):
        self.kernel = Variable(torch.FloatTensor(kernel), requires_grad=True)
        self.bias = Variable(torch.FloatTensor(bias), requires_grad=True)

    def params(self):
        return [self.kernel, self.bias]

    def forward(self, x):
        return x.mm(self.kernel) + self.bias


class ReLULayer(Layer):
    def forward(self, x):
        return x.clamp(min=0)


class SequenceLayer(Layer):
    def __init__(self, layers):
        self.layers = layers

    def params(self):
        params = []
        for layer in self.layers:
            params += layer.params()
        return params

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x


class Spec(object):
    def build(self, form=None):
        raise NotImplementedError


class DataSpec(Spec):
    def __init__(self, shape, dtype):
        self.form = Form(shape, dtype)

    def build(self, form=None):
        if form is None:
            form = self.form
        else:
            assert self.form.equals(form)
        return DataLayer(form), form


class DenseSpec(Spec):
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def build(self, form=None):
        in_dim, = form.shape
        kernel = np.random.normal(0, 1, (in_dim, self.out_dim))
        bias = np.random.normal(0, 1, (self.out_dim))
        out_shape = (self.out_dim,)
        return DenseLayer(kernel, bias), Form(out_shape, form.dtype)


class ReLUSpec(Spec):
    def build(self, form=None):
        return ReLULayer(), form


class SequenceSpec(Spec):
    def __init__(self, specs):
        self.specs = specs

    def build(self, form=None):
        layers = []
        for spec in self.specs:
            layer, form = spec.build(form)
            layers.append(layer)
        return SequenceLayer(layers), form


class Optimizer(object):
    def set_params(self, params):
        self.params = params

    def step_param(self, param):
        raise NotImplementedError

    def step(self):
        for param in self.params:
            self.step_param(param)


class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def step_param(self, param):
        param.data -= self.lr * param.grad.data
        param.grad.data.zero_()


def mean_squared_error(true, pred):
    return (true - pred).pow(2).sum()


dtype = torch.FloatTensor

N, D_in, H, D_out = 64, 1000, 100, 10

x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

model = SequenceSpec([
    DataSpec((D_in,), 'float32'),
    DenseSpec(H),
    ReLUSpec(),
    DenseSpec(D_out),
])

model, out_shape = model.build()

learning_rate = 1e-6
opt = SGD(learning_rate)
opt.set_params(model.params())

for t in range(500):
    y_pred = model.forward(x)
    loss = mean_squared_error(y, y_pred)
    print(t, loss.data[0])
    loss.backward()
    opt.step()
