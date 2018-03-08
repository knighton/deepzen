import torch
from torch.autograd import Variable


class Layer(object):
    def params(self):
        return []

    def forward(self, x):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, kernel):
        self.kernel = kernel

    def params(self):
        return [self.kernel]

    def forward(self, x):
        return x.mm(self.kernel)


class ReLU(Layer):
    def forward(self, x):
        return x.clamp(min=0)


class Sequence(Layer):
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


def mean_squared_error(true, pred):
    return (true - pred).pow(2).sum()


dtype = torch.FloatTensor

N, D_in, H, D_out = 64, 1000, 100, 10

x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)

model = Sequence([
    Dense(w1),
    ReLU(),
    Dense(w2),
])

params = model.params()

learning_rate = 1e-6
for t in range(500):
  y_pred = model.forward(x)
  
  loss = mean_squared_error(y, y_pred)
  print(t, loss.data[0])

  loss.backward()

  for param in params:
    param.data -= learning_rate * param.grad.data
    param.grad.data.zero_()
