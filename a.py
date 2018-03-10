import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F

from sunyata.dataset.mnist import load_mnist


DTYPE = torch.cuda.FloatTensor


class Backend(object):
    def shape(self, x):
        return tuple(x.size())

    def dtype_of(self, x):
        assert isinstance(x.data, DTYPE)
        return 'float32'

    def flatten(self, x):
        return x.view(x.size()[0], -1)

    def tensor(self, x):
        assert isinstance(x, np.ndarray)
        return torch.from_numpy(x).type(DTYPE)

    def constant(self, x):
        assert isinstance(x, DTYPE)
        return Variable(x, requires_grad=False)

    def variable(self, x):
        assert isinstance(x, DTYPE)
        return Variable(x, requires_grad=True)

    def matmul(self, a, b):
        return a.mm(b)

    def clip(self, x, min=-np.inf, max=np.inf):
        return x.clamp(min, max)

    def softmax(self, x):
        return F.softmax(x, -1)

    def sub(self, x, decr):
        x.data -= decr

    def grad(self, x):
        return x.grad.data

    def zero_grad(self, x):
        x.grad.data.zero_()

    def pow(self, x, power):
        return x.pow(power)

    def sum(self, x):
        return x.sum()

    def log(self, x):
        return x.log()

    def mean(self, x):
        return x.mean()

    def equal(self, a, b):
        return a == b

    def argmax(self, x, axis=-1):
        return x.max(axis)[1]


Z = Backend()


class Form(object):
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    def equals(self, other):
        return self.shape == other.shape and self.dtype == other.dtype

    def check(self, x):
        assert Z.shape(x)[1:] == self.shape
        assert Z.dtype_of(x) == self.dtype


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


class FlattenLayer(Layer):
    def forward(self, x):
        return Z.flatten(x)


class DenseLayer(Layer):
    def __init__(self, kernel, bias):
        self.kernel = Z.variable(Z.tensor(kernel))
        self.bias = Z.variable(Z.tensor(bias))

    def params(self):
        return [self.kernel, self.bias]

    def forward(self, x):
        return Z.matmul(x, self.kernel) + self.bias


class ReLULayer(Layer):
    def forward(self, x):
        return Z.clip(x, min=0)


class SoftmaxLayer(Layer):
    def forward(self, x):
        return Z.softmax(x)


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


class FlattenSpec(Spec):
    def build(self, form=None):
        out_shape = int(np.prod(form.shape)),
        return FlattenLayer(), Form(out_shape, form.dtype)


class DenseSpec(Spec):
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def build(self, form=None):
        in_dim, = form.shape
        kernel = np.random.normal(
            0, 0.1, (in_dim, self.out_dim)).astype('float32')
        bias = np.zeros(self.out_dim, 'float32')
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


class SoftmaxSpec(Spec):
    def build(self, form=None):
        return SoftmaxLayer(), form


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
        Z.sub(param, self.lr * Z.grad(param))
        Z.zero_grad(param)


def mean_squared_error(true, pred):
    return Z.sum(Z.pow(true - pred, 2))


def categorical_cross_entropy(true, pred):
    pred = Z.clip(pred, 1e-6, 1 - 1e-6)
    x = -true * Z.log(pred)
    return Z.mean(x)


def categorical_accuracy(true, pred):
    true_indices = Z.argmax(true, -1)
    pred_indices = Z.argmax(pred, -1)
    hits = Z.equal(true_indices, pred_indices)
    hits = hits.type(DTYPE)
    return hits.mean()


def each_split_batch(split, batch_size):
    x, y = split
    num_samples = len(x)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    num_batches = num_samples // batch_size
    for batch in range(num_batches):
        a = batch * batch_size
        z = (batch + 1) * batch_size
        yield x[a:z], y[a:z]


def each_dataset_batch(dataset, batch_size):
    (x_train, y_train), (x_test, y_test) = train, test = dataset
    num_train = len(x_train)
    num_test = len(x_test)
    splits = np.concatenate([np.ones(num_train), np.zeros(num_test)])
    np.random.shuffle(splits)
    each_train_batch = each_split_batch(train, batch_size)
    each_test_batch = each_split_batch(test, batch_size)
    for split in splits:
        if split:
            yield next(each_train_batch), (True,)
        else:
            yield next(each_test_batch), (False,)


class Model(object):
    def __init__(self, layer):
        self.layer = layer

    def fit_on_batch(self, optim, x, y_true):
        y_pred = self.layer.forward(x)
        loss = categorical_cross_entropy(y_true, y_pred)
        loss_value = loss.data[0]
        acc_value = categorical_accuracy(y_true, y_pred).data[0]
        loss.backward()
        optim.step()
        return loss_value, acc_value

    def fit_on_epoch(self, optim, dataset, batch_size):
        losses = []
        accs = []
        for (x, y), is_training in each_dataset_batch(dataset, batch_size):
            x = Z.constant(Z.tensor(x))
            y = Z.constant(Z.tensor(y))
            loss, acc = self.fit_on_batch(optim, x, y)
            losses.append(loss)
            accs.append(acc)
        return np.mean(np.array(losses)), np.mean(np.array(accs))

    def fit(self, optim, dataset, epochs, batch_size):
        optim.set_params(self.layer.params())
        for epoch in range(epochs):
            loss, acc = self.fit_on_epoch(optim, dataset, batch_size)
            print('Epoch %d: loss %.3f acc %.3f%%' % (epoch, loss, acc * 100))


batch_size = 64
lr = 1e-3
epochs = 10

dataset = load_mnist()
x_sample = dataset[0][0][0]
y_sample = dataset[0][1][0]
y_dim, = y_sample.shape

spec = SequenceSpec([
    DataSpec(x_sample.shape, x_sample.dtype),
    FlattenSpec(),
    DenseSpec(256),
    ReLUSpec(),
    DenseSpec(64),
    ReLUSpec(),
    DenseSpec(y_dim),
    SoftmaxSpec(),
])
layer, out_shape = spec.build()
model = Model(layer)

optim = SGD(lr)
model.fit(optim, dataset, epochs, batch_size)
