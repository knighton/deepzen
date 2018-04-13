class TrainingCursor(object):
    """
    Tracks training progress across epochs and batches.
    """

    def __init__(self, begin_epoch, end_epoch, batch_size, batches_per_epoch,
                 train_batches_per_epoch, test_batches_per_epoch, epoch, batch,
                 train_batches_done, test_batches_done):
        self.begin_epoch = begin_epoch
        self.end_epoch = end_epoch

        self.batch_size = batch_size

        self.batches_per_epoch = batches_per_epoch
        self.train_batches_per_epoch = train_batches_per_epoch
        self.test_batches_per_epoch = test_batches_per_epoch

        self.epoch = epoch
        self.batch = batch
        self.train_batches_done = train_batches_done
        self.test_batches_done = test_batches_done

    @classmethod
    def start(cls, begin_epoch, end_epoch, batch_size, batches_per_epoch,
              train_batches_per_epoch, test_batches_per_epoch):
        epoch = begin_epoch
        batch = 0
        train_batches_done = 0
        test_batches_done = 0
        return cls(begin_epoch, end_epoch, batch_size, batches_per_epoch,
                   train_batches_per_epoch, test_batches_per_epoch, epoch,
                   batch, train_batches_done, test_batches_done)

    @classmethod
    def start_from_dataset(cls, dataset, begin_epoch, end_epoch, batch_size):
        batches_per_epoch = dataset.num_batches(batch_size)
        train_batches_per_epoch = dataset.train.num_batches(batch_size)
        test_batches_per_epoch = dataset.test.num_batches(batch_size)
        return cls.start(begin_epoch, end_epoch, batch_size, batches_per_epoch,
                         train_batches_per_epoch, test_batches_per_epoch)

    def _also_completed_epoch(self):
        assert self.batch == self.batches_per_epoch
        assert self.train_batches_done == self.train_batches_per_epoch
        assert self.test_batches_done == self.test_batches_per_epoch
        self.epoch += 1
        self.batch = 0
        self.train_batches_done = 0
        self.test_batches_done = 0
        return self.epoch != self.end_epoch

    def completed_batch(self, is_training):
        if is_training:
            self.train_batches_done += 1
        else:
            self.test_batches_done += 1
        self.batch += 1
        if self.batch == self.batches_per_epoch:
            did_epoch = True
            keep_going = self._also_completed_epoch()
        else:
            did_epoch = False
            keep_going = True
        return did_epoch, keep_going
