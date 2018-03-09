import os


def get_sunyata_root():
    """
    Get the Sunyata root directory.

    That is ~/.sunyata/, unless SUNYATA_ROOT is set in the environment.

    out:
        str  root_dir  Sunyata root directory
    """
    dir_name = os.environ.get('SUNYATA_ROOT')
    if dir_name:
        return dir_name
    return os.path.expanduser('~/.sunyata/')


def get_dataset_root(dataset_name):
    """
    Get the root directory of a Sunyata dataset.

    That is (Sunyata root dir)/dataset/(dataset name).

    out:
        str  root_dir  Sunyata dataset root directory
    """
    return os.path.join(get_sunyata_root(), 'dataset', dataset_name)
