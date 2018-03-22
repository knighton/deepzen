import os


def get_root_dir():
    """
    Get the DeepZen root directory.

    That is ~/.deepzen/, unless DEEPZEN_ROOT_DIR is set in the environment.

    out:
        str  root_dir  DeepZen root directory
    """
    dir_name = os.environ.get('DEEPZEN_ROOT_DIR')
    if dir_name:
        return dir_name
    return os.path.expanduser('~/.deepzen/')


def get_dataset_root_dir(dataset_name):
    """
    Get the root directory of a DeepZen dataset.

    That is (DeepZen root dir)/dataset/(dataset name).

    out:
        str  root_dir  DeepZen dataset root directory
    """
    return os.path.join(get_root_dir(), 'dataset', dataset_name)
