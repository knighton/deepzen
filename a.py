from contextlib import contextmanager
import json
import numpy as np
import mxnet as mx
import torch
from torch.autograd import Variable
from torch.nn import functional as F

from sunyata.dataset.mnist import load_mnist


class Backend(object):
    pass


class MXNetBackend(object):
    def shape(self, x):
        return x.shape

    def dtype_of(self, x):
        return x.dtype.__name__

    def cast(self, x, dtype):
        return x.astype(dtype)

    def flatten(self, x):
        return mx.nd.flatten(x)

    def tensor(self, x):
        assert isinstance(x, np.ndarray)
        return mx.nd.array(x)

    def constant(self, x):
        assert isinstance(x, mx.nd.NDArray)
        return x.copy()

    def variable(self, x):
        assert isinstance(x, mx.nd.NDArray)
        x = x.copy()
        x.attach_grad()
        return x

    def zeros_like(self, x):
        return mx.nd.zeros_like(x)

    def ones_like(self, x):
        return mx.nd.ones_like(x)

    @contextmanager
    def autograd_record(self):
        with mx.autograd.record():
            yield

    def tensor_to_numpy(self, x):
        return x.asnumpy()

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


class PyTorchBackend(object):
    FLOAT32 = torch.FloatTensor

    def shape(self, x):
        return tuple(x.size())

    def dtype_of(self, x):
        assert isinstance(x.data, self.FLOAT32)
        return 'float32'

    def cast(self, x, dtype):
        assert dtype == 'float32'
        return x.type(self.FLOAT32)

    def flatten(self, x):
        return x.view(x.size()[0], -1)

    def tensor(self, x):
        assert isinstance(x, np.ndarray)
        return torch.from_numpy(x).type(self.FLOAT32)

    def constant(self, x):
        assert isinstance(x, self.FLOAT32)
        return Variable(x, requires_grad=False)

    def variable(self, x):
        assert isinstance(x, self.FLOAT32)
        return Variable(x, requires_grad=True)

    def zeros_like(self, x):
        if isinstance(x, Variable):
            x = x.data
        assert isinstance(x, self.FLOAT32)
        if x.is_cuda:
            return torch.zeros_like(x).cuda()
        else:
            return torch.zeros_like(x)

    def ones_like(self, x):
        if isinstance(x, Variable):
            x = x.data
        assert isinstance(x, self.FLOAT32)
        if x.is_cuda:
            return torch.ones_like(x).cuda()
        else:
            return torch.ones_like(x)

    @contextmanager
    def autograd_record(self):
        yield

    def tensor_to_numpy(self, x):
        if x.is_cuda:
            return x.cpu().numpy()
        else:
            return x.numpy()

    def constant_to_numpy(self, x):
        if x.data.is_cuda:
            return x.data.cpu().numpy()
        else:
            return x.data.numpy()

    def variable_to_numpy(self, x):
        if x.data.is_cuda:
            return x.data.cpu().numpy()
        else:
            return x.data.numpy()

    def matmul(self, a, b):
        return a.mm(b)

    def clip(self, x, min=-np.inf, max=np.inf):
        return x.clamp(min, max)

    def softmax(self, x):
        return F.softmax(x, -1)

    def assign_sub(self, x, decr):
        x.data -= decr
        x.grad.data.zero_()

    def grad(self, x):
        return x.grad.data

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


# Z = MXNetBackend()
Z = PyTorchBackend()


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


class Optimizee(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def grad(self):
        return Z.grad(self.data)


class Optimizer(object):
    def set_params(self, xx):
        self.envs = [Optimizee(**self.make_env(x)) for x in xx]

    def make_env(self, x):
        raise NotImplementedError

    def step_one(self, env):
        raise NotImplementedError

    def step(self):
        for env in self.envs:
            self.step_one(env)


class SGD(Optimizer):
    def __init__(self, lr=0.05):
        super().__init__()
        assert 0 < lr
        self.lr = lr

    def make_env(self, param):
        return {
            'data': param,
            'lr': self.lr,
        }

    def step_one(self, env):
        Z.assign_sub(env.data, env.lr * env.grad())


def momentum(momentum, old_value, new_value):
    return momentum * old_value + (1 - momentum) * new_value


class SGDM(Optimizer):
    def __init__(self, lr=0.05, momentum=0.9):
        super().__init__()
        assert 0 < lr
        assert 0 <= momentum <= 1
        self.lr = lr
        self.momentum = momentum

    def make_env(self, param):
        return {
            'data': param,
            'lr': self.lr,
            'momentum': self.momentum,
            'velocity': Z.zeros_like(param),
        }

    def step_one(self, env):
        self.velocity = momentum(
            env.momentum, env.velocity, env.lr * env.grad())
        Z.assign_sub(env.data, self.velocity)


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
    hits = Z.cast(hits, 'float32')
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
            yield next(each_train_batch), True
        else:
            yield next(each_test_batch), False


class Model(object):
    def __init__(self, layer):
        self.layer = layer

    def train_on_batch(self, optim, x, y_true):
        with Z.autograd_record():
            y_pred = self.layer.forward(x)
            loss = categorical_cross_entropy(y_true, y_pred)
        acc = categorical_accuracy(y_true, y_pred)
        loss_value = Z.variable_to_numpy(loss)[0]
        acc_value = Z.variable_to_numpy(acc)[0]
        loss.backward()
        optim.step()
        return loss_value, acc_value

    def test_on_batch(self, x, y_true):
        y_pred = self.layer.forward(x)
        loss = categorical_cross_entropy(y_true, y_pred)
        acc = categorical_accuracy(y_true, y_pred)
        loss_value = Z.variable_to_numpy(loss)[0]
        acc_value = Z.variable_to_numpy(acc)[0]
        return loss_value, acc_value

    def fit_on_epoch(self, optim, dataset, batch_size):
        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []
        for (x, y), is_training in each_dataset_batch(dataset, batch_size):
            x = Z.constant(Z.tensor(x))
            y = Z.constant(Z.tensor(y))
            if is_training:
                loss, acc = self.train_on_batch(optim, x, y)
                train_losses.append(loss)
                train_accs.append(acc)
            else:
                loss, acc = self.test_on_batch(x, y)
                test_losses.append(loss)
                test_accs.append(acc)
        train_loss = float(np.mean(train_losses))
        train_acc = float(np.mean(train_accs))
        test_loss = float(np.mean(test_losses))
        test_acc = float(np.mean(test_accs))
        return {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
        }

    def fit(self, optim, dataset, epochs, batch_size):
        optim.set_params(self.layer.params())
        for epoch in range(epochs):
            ret = self.fit_on_epoch(optim, dataset, batch_size)
            ret['epoch'] = epoch
            print(json.dumps(ret, indent=4, sort_keys=True))


batch_size = 64
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

optim = SGDM(lr=0.05, momentum=0.9)
model.fit(optim, dataset, epochs, batch_size)
