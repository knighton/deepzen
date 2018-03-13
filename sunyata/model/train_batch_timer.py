import numpy as np
from time import time


class TrainBatchTimer(object):
    def __init__(self, num_batches, num_callbacks, crit_counts):
        self.num_batches = num_batches
        self.num_callbacks = num_callbacks
        self.crit_counts = crit_counts
        self.num_losses = len(crit_counts)
        self.num_metrics = sum(crit_counts) - len(crit_counts)

        # Times per train_on_batch():
        #     1                  Begin.
        #     2 * num_callbacks  Execute "on begin" callbacks.
        #     2                  Forward propagate.
        #     2                  Compute all the losses.
        #     2 * num_losses     Compute the loss per output.
        #     2                  Compute all the additional metrics.
        #     2 * num_metrics    Compute the additional metrics per output.
        #     2                  Backpropagate gradients.
        #     2                  Optimizer step.
        #     2 * num_callbacks  Execute "on end" callbacks.
        #     1                  End.
        self.times_per_batch = \
            1 + \
            2 * num_callbacks + \
            2 + \
            2 + \
            2 * self.num_losses + \
            2 + \
            2 * self.num_metrics + \
            2 + \
            2 + \
            2 * num_callbacks + \
            1
        self.times = np.zeros((num_batches, self.times_per_batch), 'float64')
        self.batch_id = 0
        self.time_id = 0

    def begin(self):
        assert 0 <= self.batch_id < self.num_batches
        assert self.time_id == 0
        self.mark()

    def mark(self):
        self.times[self.batch_id, self.time_id] = time()
        self.time_id += 1

    def end(self):
        assert self.time_id == self.times_per_batch - 1
        self.mark()
        self.batch_id += 1
        self.time_id = 0
