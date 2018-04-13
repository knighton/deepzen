class Progress(object):
    """
    Tracks training progress across epochs and batches.
    """

    def __init__(self, begin_epoch, end_epoch, batch_size, batches_per_epoch,
                 train_batches_per_epoch, test_batches_per_epoch, current_epoch,
                 batches_done, train_batches_done, test_batches_done):
        self.begin_epoch = begin_epoch
        self.end_epoch = end_epoch

        self.batch_size = batch_size

        self.batches_per_epoch = batches_per_epoch
        self.train_batches_per_epoch = train_batches_per_epoch
        self.test_batches_per_epoch = test_batches_per_epoch

        self.current_epoch = current_epoch
        self.batches_done = batches_done
        self.train_batches_done = train_batches_done
        self.test_batches_done = test_batches_done

    @classmethod
    def init_from_batch_counts(
            cls, begin_epoch, end_epoch, batch_size, batches_per_epoch,
            train_batches_per_epoch, test_batches_per_epoch):
        current_epoch = begin_epoch
        batches_done = 0
        train_batches_done = 0
        test_batches_done = 0
        return cls(begin_epoch, end_epoch, batch_size, batches_per_epoch,
                   train_batches_per_epoch, test_batches_per_epoch,
                   current_epoch, batches_done, train_batches_done,
                   test_batches_done)

    @classmethod
    def init_from_dataset(cls, dataset, begin_epoch, end_epoch, batch_size):
        batches_per_epoch = dataset.num_batches(batch_size)
        train_batches_per_epoch = dataset.train.num_batches(batch_size)
        test_batches_per_epoch = dataset.test.num_batches(batch_size)
        return cls.init_from_batch_counts(
            begin_epoch, end_epoch, batch_size, batches_per_epoch,
            train_batches_per_epoch, test_batches_per_epoch)

    def _did_epoch(self):
        assert self.batches_done == self.batches_per_epoch
        assert self.train_batches_done == self.train_batches_per_epoch
        assert self.test_batches_done == self.test_batches_per_epoch
        self.current_epoch += 1
        self.batches_done = 0
        self.train_batches_done = 0
        self.test_batches_done = 0
        return self.current_epoch != self.end_epoch

    def did_batch(self, is_training):
        if is_training:
            self.train_batches_done += 1
        else:
            self.test_batches_done += 1
        self.batches_done += 1
        if self.batches_done == self.batches_per_epoch:
            keep_going = self._did_epoch()
        else:
            keep_going = True
        return keep_going
