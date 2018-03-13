import os
from scipy.io import loadmat

from ..nondiff.one_hot import to_one_hot
from ..util.net import download
from ..util.py import require_kwargs
from ..util.sunyata import get_dataset_root


DATASET_NAME = 'svhn'
TRAIN_URL = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
TEST_URL = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'


def _load_split(one_hot, verbose, x_dtype, y_dtype, dataset_root, split_url):
    """
    Load a SVHN split.

    in:
        bool        one_hot       Whether to one-hot the classes.
        {0, 1, 2}   verbose       Logging verbosity level.
        str         x_dtype       The desired dtype of X.
        str         y_dtype       The desired dtype of Y.
        str         dataset_root  Sunyata SVHN dataset root directory.
        str         split_url     URL to download the SVHN split from.

    out:
        np.ndarray  x             The images to feed to the neural net.
        np.ndarray  y             The classes to feed to the neural net.
    """
    split_local = os.path.join(dataset_root, os.path.basename(split_url))
    if not os.path.exists(split_local):
        download(split_url, split_local, verbose)
    d = loadmat(split_local)
    x = d['X']
    x = x.transpose((3, 2, 0, 1))
    x = x.astype(x_dtype)
    x /= 127.5
    x -= 1
    y = d['y']
    y = y.squeeze()
    y %= 10
    if one_hot:
        y = to_one_hot(y, 10, y_dtype)
    else:
        y = y.astype(y_dtype)
    return x, y
    

@require_kwargs
def load_svhn(dataset_name=DATASET_NAME, one_hot=True, test_url=TEST_URL,
              train_url=TRAIN_URL, verbose=2, x_dtype='float32',
              y_dtype='float32'):
    """
    Load the Street View House Numbers (SVHN) dataset.

    in:
        str        dataset_name  Sunyata dataset directory name.
        bool       one_hot       Whether to one-hot the classes.
        str        test_url      URL to download the test split from.
        str        train_url     URL to download the training split from.
        {0, 1, 2}  verbose       Logging verbosity level.
        str        x_dtype       The desired dtype of X.
        str        y_dtype       The desired dtype of Y.

    out:
        tuple      dataset       The dataset splits as numpy ndarrays.
        list<str>  class_names   The list of string class names.
    """
    dataset_root = get_dataset_root(dataset_name)
    train = _load_split(one_hot, verbose, x_dtype, y_dtype, dataset_root,
                        train_url)
    test = _load_split(one_hot, verbose, x_dtype, y_dtype, dataset_root,
                       test_url)
    class_names = list('0123456789')
    return (train, test),  class_names
