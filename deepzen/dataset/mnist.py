import gzip
import numpy as np
import os
import pickle

from ..transform.one_hot import to_one_hot
from ..util.config import get_dataset_root_dir
from ..util.net import download
from ..util.py import require_kwargs


DATASET_NAME = 'mnist'
URL = 'https://s3.amazonaws.com/img-datasets/mnist.pkl.gz'


def _transform_x(x, dtype):
    """
    Normalize image data for feeding to the neural net.

    in:
        np.ndarray  x      Raw pixels.
        str         dtype  Desired dtype.

    out:
        np.ndarray  x      Floats from -1 to +1.
    """
    x = np.expand_dims(x, 1)
    x = x.astype(dtype)
    x /= 127.5
    x -= 1
    return x


def _transform_y(y, one_hot, dtype):
    """
    Transform classes for feeding to the neural net.

    in:
        np.ndarray  y            Classes as integer IDs.
        bool        one_hot      Whether to one-hot the classes.
        int         num_classes  Number of classes (for one-hot).
        str         dtype        Desired dtype.

    out:
        np.ndarray  y            Transformed class values.
    """
    if one_hot:
        y = to_one_hot(y, 10, dtype)
    else:
        y = y.astype(dtype)
    return y


@require_kwargs
def load_mnist(dataset_name=DATASET_NAME, one_hot=True, url=URL, verbose=2,
               x_dtype='float32', y_dtype='float32'):
    """
    Load the MNIST handwritten digits dataset.

    in:
        str        dataset_name  DeepZen dataset directory name.
        bool       one_hot       Whether to one-hot the classes.
        str        url           URL to download raw data from.
        {0, 1, 2}  verbose       Logging verbosity level.
        str        x_dtype       The desired dtype of X.
        str        y_dtype       The desired dtype of Y.

    out:
        tuple      dataset       The dataset splits as numpy ndarrays.
        list<str>  class_names   The list of string class names.
    """
    dataset_root = get_dataset_root_dir(dataset_name)
    local = os.path.join(dataset_root, os.path.basename(url))
    if not os.path.exists(local):
        download(url, local, verbose)
    (x_train, y_train), (x_val, y_val) = \
        pickle.load(gzip.open(local), encoding='latin1')
    x_train = _transform_x(x_train, x_dtype)
    y_train = _transform_y(y_train, one_hot, y_dtype)
    x_val = _transform_x(x_val, x_dtype)
    y_val = _transform_y(y_val, one_hot, y_dtype)
    dataset = (x_train, y_train), (x_val, y_val)
    class_names = list('0123456789')
    return dataset, class_names
