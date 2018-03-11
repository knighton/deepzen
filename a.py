import json
import numpy as np

from sunyata import backend as Z
from sunyata.dataset.mnist import load_mnist
from sunyata.iter.dataset import Dataset
from sunyata.iter.ram_split import RamSplit
from sunyata.layer import *  # noqa
from sunyata.optim import *  # noqa
from sunyata.crit.loss import categorical_cross_entropy
from sunyata.crit.metric import categorical_accuracy


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
        for (xx, yy), is_training in dataset.each_batch(batch_size):
            x, = xx
            y, = yy
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
        (x_train, y_train), (x_test, y_test) = dataset
        train = RamSplit(x_train, y_train)
        test = RamSplit(x_test, y_test)
        dataset = Dataset(train, test)
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
