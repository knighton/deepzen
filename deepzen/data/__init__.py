from .dataset import Dataset
from .ram_split import RamSplit
from .split import Split


def unpack_split(split):
    if isinstance(split, Split):
        return split

    xx, yy = split
    return RamSplit(xx, yy)


def unpack_dataset(dataset, test_frac=None):
    if isinstance(dataset, Dataset):
        assert test_frac is None
        return dataset

    if test_frac is not None:
        assert False, 'TODO: Perform train/test split.'

    train, test = dataset
    train = unpack_split(train)
    test = unpack_split(test)
    return Dataset(train, test)
