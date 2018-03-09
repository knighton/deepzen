import gzip
import numpy as np
import os
import pickle

from ..nondiff.one_hot import to_one_hot
from ..util.net import download
from ..util.py import require_kwargs
from ..util.sunyata import get_dataset_root


DATASET_NAME = 'mnist'
URL = 'https://s3.amazonaws.com/img-datasets/mnist.pkl.gz'


def _transform_x(x, dtype):
    x = np.expand_dims(x, 1)
    x = x.astype(dtype)
    x /= 127.5
    x -= 1
    return x


def _transform_y(y, one_hot, dtype):
    if one_hot:
        y = to_one_hot(y, 10, dtype)
    else:
        y = y.astype(dtype)
    return y


@require_kwargs
def load_mnist(dataset_name=DATASET_NAME, one_hot=True, x_dtype='float32',
               y_dtype='float32', url=URL, verbose=2):
    """
    Load the MNIST handwritten digits dataset.

    in:
        str    dataset_name  Sunyata dataset directory name
        bool   one_hot       Whether to one_hot the classes
        str    x_dtype       The desired dtype of X
        str    y_dtype       The desired dtype of Y
        str    url           URL to download raw data from
        int    verbose       Verbosity level

    out:
        tuple  dataset      The dataset splits
    """
    dataset_dir = get_dataset_root(dataset_name)
    local = os.path.join(dataset_dir, os.path.basename(url))
    if not os.path.exists(local):
        download(url, local, verbose)
    (x_train, y_train), (x_val, y_val) = \
        pickle.load(gzip.open(local), encoding='latin1')
    x_train = _transform_x(x_train, x_dtype)
    y_train = _transform_y(y_train, one_hot, y_dtype)
    x_val = _transform_x(x_val, x_dtype)
    y_val = _transform_y(y_val, one_hot, y_dtype)
    return (x_train, y_train), (x_val, y_val)
