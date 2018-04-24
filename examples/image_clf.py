from argparse import ArgumentParser

from deepzen.app import *  # noqa
from deepzen.node import *  # noqa
from deepzen.task.cifar import load_cifar
from deepzen.task.mnist import load_mnist
from deepzen.task.svhn import load_svhn


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('--task', type=str, default='mnist',
                    help='The task to train on.')
    ap.add_argument('--model', type=str, default='simple',
                    help='The model architecture to train.')
    ap.add_argument('--optimizer', type=str, default='adam', help='Optimizer.')
    ap.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    ap.add_argument('--begin_epoch', type=int, default=0, help='Start epoch.')
    ap.add_argument('--end_epoch', type=int, default=100, help='Stop epoch.')
    ap.add_argument('--spy', type=str, default='server,progress_bar,rows',
                    help='List of training monitors.')
    return ap.parse_args()


class Datasets(object):
    """
    A collection of image classification tasks.

    Select with --task.
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
        return Input(image_shape, dtype) > Flatten > Dense(128) > ReLU > \
            Dense(num_classes) > Softmax

    @classmethod
    def mlp(cls, image_shape, dtype, num_classes):
        block = lambda dim: Dense(dim) > BatchNorm > ReLU > Dropout
        return Input(image_shape, dtype) > Flatten > block(512) > block(128) > \
            Dense(num_classes) > Softmax

    @classmethod
    def cnn(cls, image_shape, dtype, num_classes):
        block = lambda dim: \
            Conv(dim) > BatchNorm > ReLU > Dropout(0.25) > MaxPool(2)
        return Input(image_shape, dtype) > block(16) > block(16) > block(16) > \
            block(16) > Flatten > Dense(num_classes) > Softmax

    @classmethod
    def vgg16(cls, image_shape, dtype, num_classes):
        return Input(image_shape, dtype) > \
            VGG16(include_top=True, classes=num_classes)

    @classmethod
    def vgg19(cls, image_shape, dtype, num_classes):
        return Input(image_shape, dtype) > \
            VGG19(include_top=True, classes=num_classes)

    @classmethod
    def get(cls, name, dataset):
        x_train, y_train = dataset[0]
        image = x_train[0]
        num_classes = len(y_train[0])
        get = getattr(cls, name)
        return get(image.shape, image.dtype.name, num_classes)


def run(args):
    dataset, class_names = Datasets.get(args.task)
    model = Models.get(args.model, dataset)
    model.fit_clf(dataset, optimizer=args.optimizer, batch_size=args.batch_size,
                  begin_epoch=args.begin_epoch, end_epoch=args.end_epoch,
                  spy=args.spy)


if __name__ == '__main__':
    run(parse_args())
