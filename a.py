import numpy as np
import torch
from torch.autograd import Variable


class Layer(object):
    def params(self):
        return []

    def forward(self, x):
        raise NotImplementedError


class DenseLayer(Layer):
    def __init__(self, kernel):
        self.kernel = Variable(torch.FloatTensor(kernel), requires_grad=True)

    def params(self):
        return [self.kernel]

    def forward(self, x):
        return x.mm(self.kernel)


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
    def build(self):
        raise NotImplementedError


class DenseSpec(Spec):
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim

    def build(self):
        kernel = np.random.normal(0, 1, (self.in_dim, self.out_dim))
        return DenseLayer(kernel)


class ReLUSpec(Spec):
    def build(self):
        return ReLULayer()


class SequenceSpec(Spec):
    def __init__(self, specs):
        self.specs = specs

    def build(self):
        layers = []
        for spec in self.specs:
            layer = spec.build()
            layers.append(layer)
        return SequenceLayer(layers)


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
    DenseSpec(D_in, H),
    ReLUSpec(),
    DenseSpec(H, D_out),
])

model = model.build()

learning_rate = 1e-6
opt = SGD(learning_rate)
opt.set_params(model.params())

for t in range(500):
  y_pred = model.forward(x)
  
  loss = mean_squared_error(y, y_pred)
  print(t, loss.data[0])

  loss.backward()

  opt.step()
