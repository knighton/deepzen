from argparse import ArgumentParser

from deepzen.dataset.cifar import load_cifar
from deepzen.dataset.mnist import load_mnist
from deepzen.dataset.svhn import load_svhn
from deepzen.node import *  # noqa


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('--dataset', type=str, default='mnist',
                    help='The dataset to train on.')
    ap.add_argument('--model', type=str, default='simple',
                    help='The model architecture to train.')
    ap.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train for.')
    ap.add_argument('--batch', type=int, default=128, help='Batch size.')
    ap.add_argument('--optim', type=str, default='adam', help='Optimizer.')
    ap.add_argument('--hook', type=str, default='server,progress_bar,rows',
                    help='List of training monitors.')
    return ap.parse_args()


class Datasets(object):
    """
    A collection of image classification datasets.

    Select with --dataset.
    """

    mnist = load_mnist
    cifar10 = lambda: load_cifar(classes=10)
    cifar20 = lambda: load_cifar(classes=20)
    cifar100 = lambda: load_cifar(classes=100)
    svhn = load_svhn

    @classmethod
    def get(cls, name):
        get = getattr(cls, name)
        return get()


class Models(object):
    """
    A collection of image classification models.

    Select with --model.
    """

    @classmethod
    def simple(cls, image_shape, dtype, num_classes):
        return Data(image_shape, dtype) > Flatten > Dense(128) > ReLU > \
            Dense(num_classes) > Softmax

    @classmethod
    def mlp(cls, image_shape, dtype, num_classes):
        block = lambda dim: Dense(dim) > BatchNorm > ReLU > Dropout
        return Data(image_shape, dtype) > Flatten > block(512) > block(128) > \
            Dense(num_classes) > Softmax

    @classmethod
    def cnn(cls, image_shape, dtype, num_classes):
        block = lambda dim: \
            Conv(dim) > BatchNorm > ReLU > Dropout(0.25) > MaxPool(2)
        return Data(image_shape, dtype) > block(16) > block(16) > block(16) > \
            block(16) > Flatten > Dense(num_classes) > Softmax

    @classmethod
    def get(cls, name, dataset):
        x_train, y_train = dataset[0]
        image = x_train[0]
        num_classes = len(y_train[0])
        get = getattr(cls, name)
        return get(image.shape, image.dtype.name, num_classes)


def run(args):
    dataset, class_names = Datasets.get(args.dataset)
    model = Models.get(args.model, dataset)
    model.fit_clf(dataset, epochs=args.epochs, batch=args.batch,
                  optim=args.optim, hook=args.hook)


if __name__ == '__main__':
    run(parse_args())
