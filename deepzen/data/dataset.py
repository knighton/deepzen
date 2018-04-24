import numpy as np

from .split import Split


class Dataset(object):
    def __init__(self, train, test):
        assert isinstance(train, Split)
        if test is not None:
            assert isinstance(test, Split)
            assert train.sample_shapes == test.sample_shapes
            assert train.dtypes == test.dtypes
        self.train = train
        self.test = test

        if test:
            self.samples_per_epoch = \
                train.samples_per_epoch + test.samples_per_epoch
        else:
            self.samples_per_epoch = train.samples_per_epoch

        self.sample_shapes = train.sample_shapes
        self.x_sample_shapes = train.x_sample_shapes
        self.y_sample_shapes = train.y_sample_shapes

        self.dtypes = train.dtypes
        self.x_dtypes = train.x_dtypes
        self.y_dtypes = train.y_dtypes

    def batches_per_epoch(self, batch_size):
        batches_per_epoch = self.train.batches_per_epoch(batch_size)
        if self.test:
            batches_per_epoch += self.test.batches_per_epoch(batch_size)
        return batches_per_epoch

    def get_batch(self, batch_size, is_training, index):
        if is_training:
            split = self.train
        else:
            split = self.test
        return split.get_batch(batch_size, index)

    def shuffle(self, batch_size):
        num_train_batches = self.train.batches_per_epoch(batch_size)
        if self.test:
            num_test_batches = self.test.batches_per_epoch(batch_size)
        else:
            num_test_batches = 0
        train_batches = np.arange(num_train_batches)
        test_batches = np.arange(num_test_batches)
        x = np.zeros((num_train_batches + num_test_batches, 2), 'int64')
        x[train_batches, 0] = 1
        x[train_batches, 1] = train_batches
        x[num_train_batches + test_batches, 1] = test_batches
        np.random.shuffle(x)
        return x

    def each_batch(self, batch_size):
        for is_training, index in self.shuffle(batch_size):
            xx, yy = self.get_batch(batch_size, is_training, index)
            yield is_training, xx, yy

    def each_batch_forever(self, batch_size):
        while True:
            for batch in self.each_batch(batch_size):
                yield batch
