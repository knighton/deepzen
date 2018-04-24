from deepzen.data import unpack_dataset
from deepzen.task.cifar import load_cifar
import json
import numpy as np
from time import time
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from torch import optim

from .common import EpochResults


class MLP(nn.Module):
    def __init__(self, image_shape, image_dtype, num_classes):
        super().__init__()
        in_dim = int(np.prod(image_shape))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, num_classes)#,
        ).cuda()

    def forward_one(self, image):
        x = image.view(image.shape[0], -1)
        return self.mlp(x)

    def forward(self, xx):
        x, = xx
        y = self.forward_one(x)
        return [y]

    def train_on_batch(self, xx, yy_true, optimizer, losses, aux_meter_lists):
        self.train()

        self.zero_grad()

        yy_pred = self.forward(xx)

        costs = []
        for loss, y_true, y_pred in zip(losses, yy_true, yy_pred):
            cost = loss(y_pred, y_true)
            costs.append(cost)

        grads = [torch.ones(1).cuda() for x in costs]
        torch.autograd.backward(costs, grads)

        costs = [float(cost.cpu().data.numpy()[0]) for cost in costs]

        optimizer.step()

        aux_metric_lists = []
        for meters, y_true, y_pred in zip(aux_meter_lists, yy_true, yy_pred):
            metrics = []
            for meter in meters:
                metric = meter(y_pred, y_true)
                metric = float(metric.cpu().data.numpy()[0])
                metrics.append(metric)
            aux_metric_lists.append(metrics)

        return costs, aux_metric_lists

    def test_on_batch(self, xx, yy_true, losses, aux_meter_lists):
        self.eval()

        yy_pred = self.forward(xx)

        costs = []
        for loss, y_true, y_pred in zip(losses, yy_true, yy_pred):
            cost = loss(y_pred, y_true)
            costs.append(cost)

        costs = [float(cost.cpu().data.numpy()[0]) for cost in costs]

        aux_metric_lists = []
        for meters, y_true, y_pred in zip(aux_meter_lists, yy_true, yy_pred):
            metrics = []
            for meter in meters:
                metric = meter(y_pred, y_true)
                metric = float(metric.cpu().data.numpy()[0])
                metrics.append(metric)
            aux_metric_lists.append(metrics)

        return costs, aux_metric_lists

    def fit_on_batch(self, is_training, xx, yy_true, optimizer, losses,
                     aux_meter_lists):
        numpy_to_constant = lambda x: \
            Variable(torch.from_numpy(x).cuda(), requires_grad=False)
        xx = [numpy_to_constant(x) for x in xx]
        yy_true = [numpy_to_constant(y) for y in yy_true]
        if is_training:
            ret = self.train_on_batch(xx, yy_true, optimizer, losses,
                                      aux_meter_lists)
        else:
            ret = self.test_on_batch(xx, yy_true, losses, aux_meter_lists)
        return ret


def load_cifar10():
    return load_cifar(classes=10)


class Accuracy(object):
    def __call__(self, pred, true):
        return (pred.max(-1)[1] == true).type_as(pred).mean()


def main():
    num_epochs = 20
    batch_size = 64

    dataset, class_names = load_cifar()

    (x_train, y_train), _ = dataset
    image_shape = x_train[0].shape
    image_dtype = x_train.dtype.name
    num_classes = len(y_train[0])
    model = MLP(image_shape, image_dtype, num_classes)

    dataset = unpack_dataset(dataset)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    losses = [nn.CrossEntropyLoss()]
    aux_meter_lists = [[Accuracy()]]

    for epoch in range(num_epochs):
        results = EpochResults()
        for (xx, yy_true), is_training in dataset.each_batch(batch_size):
            classes, = yy_true
            classes = classes.argmax(-1)
            yy_true = [classes]
            t0 = time()
            costs, aux_metric_lists = model.fit_on_batch(
                is_training, xx, yy_true, optimizer, losses, aux_meter_lists)
            t = time() - t0
            results.add(is_training, costs, aux_metric_lists, t)
        print(json.dumps(results.summary(), sort_keys=True, indent=4))


if __name__ == '__main__':
    main()
