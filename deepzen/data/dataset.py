import numpy as np

from .split import Split


class Dataset(object):
    def __init__(self, train, test):
        assert isinstance(train, Split)
        assert isinstance(test, Split)
        assert train.shapes() == test.shapes()
        assert train.dtypes() == test.dtypes()
        self.train = train
        self.test = test

    def num_samples(self):
        num_samples = self.train.num_samples()
        if self.test:
            num_samples += self.test.num_samples()
        return num_samples

    def num_batches(self, batch_size):
        num_batches = self.train.num_batches(batch_size)
        if self.test:
            num_batches += self.test.num_batches(batch_size)
        return num_batches

    def shapes(self, batch_size=None):
        return self.train.shapes(batch_size)

    def dtypes(self):
        return self.train.dtypes()

    def each_batch(self, batch_size):
        num_train = self.train.num_batches(batch_size)
        num_test = self.test.num_batches(batch_size)
        splits = np.concatenate([np.ones(num_train), np.zeros(num_test)])
        np.random.shuffle(splits)
        each_train = self.train.each_batch(batch_size)
        each_test = self.test.each_batch(batch_size)
        for split in splits:
            if split:
                yield next(each_train), True
            else:
                yield next(each_test), False

    def each_batch_forever(self, batch_size):
        while True:
            for batch in self.each_batch(batch_size):
                yield batch
