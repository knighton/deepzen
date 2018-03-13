from argparse import ArgumentParser

from sunyata.dataset.cifar import load_cifar
from sunyata.dataset.mnist import load_mnist
from sunyata.dataset.svhn import load_svhn
from sunyata.layer import *  # noqa
from sunyata.model import Model


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('--dataset', type=str, default='mnist',
                    help='The dataset to train on.')
    return ap.parse_args()


def load_dataset(name):
    if name == 'mnist':
        get = lambda: load_mnist()
    elif name == 'cifar-10':
        get = lambda: load_cifar(classes=10)
    elif name == 'cifar-20':
        get = lambda: load_cifar(classes=20)
    elif name == 'cifar-100':
        get = lambda: load_cifar(classes=100)
    elif name == 'svhn':
        get = lambda: load_svhn()
    else:
        assert False
    return get()


def run(args):
    dataset, class_names = load_dataset(args.dataset)
    x_sample = dataset[0][0][0]
    y_sample = dataset[0][1][0]
    num_classes, = y_sample.shape
    spec = SequenceSpec([
        DataSpec(x_sample.shape, x_sample.dtype),
        FlattenSpec(),
        DenseSpec(256),
        ReLUSpec(),
        DenseSpec(64),
        ReLUSpec(),
        DenseSpec(num_classes),
        SoftmaxSpec(),
    ])
    model = Model(spec)
    model.fit_clf(dataset, callback='server,progress_bar,row_per_epoch')


if __name__ == '__main__':
    run(parse_args())
