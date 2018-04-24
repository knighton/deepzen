import json
import numpy as np
from time import time

from deepzen.data import unpack_dataset
from deepzen.meter import CategoricalAccuracy, CategoricalCrossEntropy
from deepzen.node import *  # noqa
from deepzen.optim import SGDM
from deepzen.task.cifar import load_cifar

from .common import EpochResults


def load_cifar10():
    return load_cifar(classes=10)


def make_mlp(image_shape, image_dtype, num_classes):
    block = lambda dim: Dense(dim) > BatchNorm > ReLU > Dropout
    return Input(image_shape, image_dtype) > Flatten > block(128) * 2 > \
        Dense(num_classes) > Softmax


def init(load_dataset, make_model):
    dataset, class_names = load_dataset()

    (x_train, y_train), _ = dataset
    image_shape = x_train[0].shape
    image_dtype = x_train.dtype.name
    num_classes = len(y_train[0])

    model = make_model(image_shape, image_dtype, num_classes)
    model.build()

    dataset = unpack_dataset(dataset)

    return dataset, class_names, model


def main():
    num_epochs = 20
    batch_size = 64

    dataset, class_names, model = init(load_cifar10, make_mlp)

    optimizer = SGDM(lr=0.001, momentum=0.9)
    optimizer.set_params(model.params())

    losses = [CategoricalCrossEntropy()]
    aux_meter_lists = [[CategoricalAccuracy()]]

    for epoch in range(num_epochs):
        results = EpochResults()
        for (xx, yy_true), is_training in dataset.each_batch(batch_size):
            t0 = time()
            costs, aux_metric_lists = model.fit_on_batch(
                is_training, xx, yy_true, optimizer, losses, aux_meter_lists)
            t = time() - t0
            results.add(is_training, costs, aux_metric_lists, t)
        print(json.dumps(results.summary(), sort_keys=True, indent=4))


if __name__ == '__main__':
    main()
