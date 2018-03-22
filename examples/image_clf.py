from argparse import ArgumentParser

from deepzen.dataset.cifar import load_cifar
from deepzen.dataset.mnist import load_mnist
from deepzen.dataset.svhn import load_svhn
from deepzen.layer import *  # noqa
from deepzen.model import Model


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('--dataset', type=str, default='mnist',
                    help='The dataset to train on.')
    ap.add_argument('--model', type=str, default='simple',
                    help='The model architecture to train.')
    ap.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train for.')
    ap.add_argument('--batch', type=int, default=128, help='Batch size.')
    ap.add_argument('--optim', type=str, default='sgd', help='Optimizer.')
    ap.add_argument('--view', type=str,
                    default='server,progress_bar,row_per_epoch',
                    help='List of training monitors.')
    return ap.parse_args()


class Datasets(object):
    """
    A collection of image classification datasets.

    Select with --dataset.
    """

    mnist = load_mnist
    cifar_10 = lambda: load_cifar(classes=10)
    cifar_20 = lambda: load_cifar(classes=20)
    cifar_100 = lambda: load_cifar(classes=100)
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
        return SequenceSpec([
            DataSpec(image_shape, dtype),
            FlattenSpec(),
            DenseSpec(128),
            ReLUSpec(),
            DenseSpec(num_classes),
            SoftmaxSpec(),
        ])

    @classmethod
    def mlp(cls, image_shape, dtype, num_classes):
        return SequenceSpec([
            DataSpec(image_shape, dtype),
            FlattenSpec(),

            DenseSpec(512),
            MovAvgBatchNormSpec(),
            ReLUSpec(),
            DropoutSpec(),

            DenseSpec(256),
            MovAvgBatchNormSpec(),
            ReLUSpec(),
            DropoutSpec(),

            DenseSpec(128),
            MovAvgBatchNormSpec(),
            ReLUSpec(),
            DropoutSpec(),

            DenseSpec(num_classes),
            SoftmaxSpec(),
        ])

    @classmethod
    def get(cls, name, dataset):
        x_train, y_train = dataset[0]
        image = x_train[0]
        num_classes = len(y_train[0])
        get = getattr(cls, name)
        spec = get(image.shape, image.dtype.name, num_classes)
        return Model(spec)


def run(args):
    dataset, class_names = Datasets.get(args.dataset)
    model = Models.get(args.model, dataset)
    model.fit_clf(dataset, epochs=args.epochs, batch=args.batch,
                  optim=args.optim, view=args.view)


if __name__ == '__main__':
    run(parse_args())
