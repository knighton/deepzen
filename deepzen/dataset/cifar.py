import numpy as np
import os
import pickle
import tarfile
from tqdm import tqdm

from ..transform.one_hot import to_one_hot
from ..util.dataset import train_test_split
from ..util.net import download
from ..util.py import require_kwargs
from ..util.config import get_dataset_root_dir


DATASET_NAME = 'cifar'
CIFAR10_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
CIFAR100_URL = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'


def _transform_x(x, dtype):
    """
    Normalize image data for feeding to the neural net.

    in:
        np.ndarray  x      Raw pixels.
        str         dtype  Desired dtype.

    out:
        np.ndarray  x      Floats from -1 to +1.
    """
    x = x.reshape(-1, 3, 32, 32).astype(dtype)
    x /= 127.5
    x -= 1
    return x


def _transform_y(y, one_hot, num_classes, dtype):
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
        y = np.array(y, 'int32')
        y = to_one_hot(y, num_classes, dtype)
    else:
        y = np.array(y, dtype)
    return y


def _load_cifar10_data(tar, one_hot, x_dtype, y_dtype, verbose):
    """
    Load the images and classes of the CIFAR-10 dataset.

    in:
        tarfile    tar      The tarfile to read.
        bool       one_hot  Whether to one-hot the the classes.
        str        x_dtype  The desired dtype of X.
        str        y_dtype  The desired dtype of Y.
        {0, 1, 2}  verbose  Logging verbosity level.

    out:
        tuple      dataset  The dataset (a single split).
    """
    if verbose == 2:
        bar = tqdm(total=5, leave=False)
    xx = []
    yy = []
    for info in tar.getmembers():
        if not info.isreg():
            continue
        if not info.path.startswith('cifar-10-batches-py/data_batch_'):
            continue
        data = tar.extractfile(info).read()
        obj = pickle.loads(data, encoding='bytes')
        x = obj[b'data']
        x = _transform_x(x, x_dtype)
        y = obj[b'labels']
        y = _transform_y(y, one_hot, 10, y_dtype)
        xx.append(x)
        yy.append(y)
        if verbose == 2:
            bar.update(1)
    if verbose == 2:
        bar.close()
    x = np.vstack(xx)
    y = np.vstack(yy)
    return x, y


def _load_cifar10_class_names(tar):
    """
    Load CIFAR-100 class names from a tar file.

    in:
        tarfile    tar     The tar file to read.

    out:
        list<str>  labels  The string name of each class.
    """
    path = 'cifar-10-batches-py/batches.meta'
    data = tar.extractfile(path).read()
    obj = pickle.loads(data, encoding='bytes')
    labels = obj[b'label_names']
    return list(map(lambda s: s.decode('utf-8'), labels))


@require_kwargs
def load_cifar10(dataset_name=DATASET_NAME, one_hot=True, url=CIFAR10_URL,
                 verbose=2, x_dtype='float32', y_dtype='float32'):
    """
    Load the CIFAR-10 image classificaiton dataset.

    in:
        str         dataset_name  DeepZen dataset directory name.
        bool        one_hot       Whether to one-hot the classes.
        str         url           URL to download CIFAR-10 dataset from.
        {0, 1, 2}   verbose       Logging verbosity level.
        str         x_dtype       The desired dtype of X.
        str         y_dtype       The desired dtype of Y.

    out:
        tuple       dataset       The dataset splits as numpy ndarrays.
    """
    dataset_root = get_dataset_root_dir(dataset_name)
    local = os.path.join(dataset_root, os.path.basename(url))
    if not os.path.exists(local):
        download(url, local, verbose)
    tar = tarfile.open(local, 'r:gz')
    x, y = _load_cifar10_data(tar, one_hot, x_dtype, y_dtype, verbose)
    class_names = _load_cifar10_class_names(tar)
    tar.close()
    return (x, y), class_names


def _load_cifar100_split(tar, classes, one_hot, x_dtype, y_dtype, split):
    """
    Load one split of the CIFAR-100 dataset.

    in:
        tarfile    tar      The tarfile to read.
        {20, 100}  classes  Whether to load coarse or fine labels.
        bool       one_hot  Whether to one-hot the the classes.
        str        x_dtype  The desired dtype of X.
        str        y_dtype  The desired dtype of Y.
        str        split    Name of the split (train or test).

    out:
        tuple      split    One split of the CIFAR-100 dataset (x, y arrays).
    """
    path = 'cifar-100-python/%s' % split
    data = tar.extractfile(path).read()
    obj = pickle.loads(data, encoding='bytes')
    x = obj[b'data']
    x = _transform_x(x, x_dtype)
    if classes == 20:
        key = b'coarse_labels'
    elif classes == 100:
        key = b'fine_labels'
    else:
        assert False
    y = obj[key]
    y = _transform_y(y, one_hot, classes, y_dtype)
    return x, y


def _load_cifar100_class_names(tar, classes):
    """
    Load CIFAR-100 class names from a tar file.

    in:
        tarfile    tar      The tarfile to read.
        {20, 100}  classes  Whether to load coarse or fine labels.

    out:
        list<str>  labels   The string name of each class.
    """
    info = tar.getmember('cifar-100-python/meta')
    data = tar.extractfile(info).read()
    obj = pickle.loads(data, encoding='bytes')
    if classes == 20:
        key = b'coarse_label_names'
    elif classes == 100:
        key = b'fine_label_names'
    else:
        assert False
    labels = obj[key]
    return list(map(lambda s: s.decode('utf-8'), labels))


@require_kwargs
def load_cifar100(classes=100, dataset_name=DATASET_NAME, one_hot=True,
                  url=CIFAR100_URL, verbose=2, x_dtype='float32',
                  y_dtype='float32'):
    """
    Load the CIFAR-20/100 image classification dataset.

    in:
        {20, 100}   classes       Number of classes (selects coarse or fine
                                  labels).
        str         dataset_name  DeepZen dataset directory name.
        bool        one_hot       Whether to one-hot the classes.
        str         url           URL to download CIFAR-20/100 dataset from.
        {0, 1, 2}   verbose       Logging verbosity level.
        str         x_dtype       The desired dtype of X.
        str         y_dtype       The desired dtype of Y.

    out:
        tuple       dataset       The dataset splits as numpy ndarrays.
        list<str>   class_names   The list of string class names.
    """

    dataset_root = get_dataset_root_dir(dataset_name)
    local = os.path.join(dataset_root, os.path.basename(url))
    if not os.path.exists(local):
        download(url, local, verbose)
    tar = tarfile.open(local, 'r:gz')
    train = _load_cifar100_split(tar, classes, one_hot, x_dtype, y_dtype,
                                 'train')
    test = _load_cifar100_split(tar, classes, one_hot, x_dtype, y_dtype, 'test')
    class_names = _load_cifar100_class_names(tar, classes)
    tar.close()
    return (train, test), class_names


@require_kwargs
def load_cifar(cifar100_url=CIFAR100_URL, cifar10_url=CIFAR10_URL, classes=10,
               dataset_name=DATASET_NAME, one_hot=True, test_frac=0.2,
               verbose=2, x_dtype='float32', y_dtype='float32'):
    """
    Load the CIFAR-10, CIFAR-20, or CIFAR-100 image classification dataset.

    in:
        str            cifar100_url  URL to download CIFAR-20/100 dataset from.
        str            cifar10_url   URL to download CIFAR-10 dataset from.
        {10, 20, 100}  classes       Number of classes (selects dataset).
        str            dataset_name  DeepZen dataset directory name.
        bool           one_hot       Whether to one-hot the classes.
        float          test_frac     If CIFAR-10, the fraction of the dataset to
                                     use for the test split.
        {0, 1, 2}      verbose       Logging verbosity level.
        str            x_dtype       The desired dtype of X.
        str            y_dtype       The desired dtype of Y.

    out:
        tuple          dataset       The dataset splits as numpy ndarrays.
        list<str>      class_names   The list of string class names.
    """
    if classes == 10:
        (x, y), class_names = load_cifar10(
            dataset_name=dataset_name, one_hot=one_hot, url=cifar10_url,
            verbose=verbose, x_dtype=x_dtype, y_dtype=y_dtype)
        train, test = train_test_split(x, y, test_frac)
    elif classes in {20, 100}:
        (train, test), class_names = load_cifar100(
            classes=classes, dataset_name=dataset_name, one_hot=one_hot,
            url=cifar100_url, verbose=verbose, x_dtype=x_dtype, y_dtype=y_dtype)
    else:
        assert False
    return (train, test), class_names
