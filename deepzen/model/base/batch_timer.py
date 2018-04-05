import numpy as np
from time import time


class SplitOnBatchTimer(object):
    """
    Efficient fine-grained intra-batch timing statistics during Model training.

    Abstract base class.  There are two concrete subclasses:
    * TrainOnBatchTimer for train_on_batch().
    * TestOnBatchTimer for test_on_batch().

    Shows you where time is being spent during training at a very fine-grained
    level.  Times forward/backward/optim, individual hook callbacks, each loss
    function and every other scorer, and everything else of note.

    Internally, it is just a cache of time.time() values in a float64 numpy
    ndarray.  Each row is the exact times of one batch's events.  Each column is
    a specific event, which all happen in a fixed order.

    During execution, it fills the rows of the cache in ascending order, then
    switches to replacing rows randomly with new timing data.  Rows are filled
    each batch by calling start(), then mark() for a precalculated number of
    times, then stop().

    When requested, it pairs up the columns gathering how long each event took,
    then returns timing summary statistics and quantiles for each event.  You
    should call fit() with the 'server' hook to visualize this data in real time
    with human-friendly charts, otherwise the granularity can be unwieldy.
    """

    def init_middle(self, middle_offset):
        raise NotImplementedError

    def __init__(self, cache_size, hook_names, scorer_name_lists):
        # During execution, we will fill the rows of the cache in ascending
        # order until full, then start replacing rows at random.
        self.cache_used = 0
        self.cache_size = cache_size

        # The names of the hooks and scorers (losses and auxiliary scorers).
        #
        # The names are used for displaying results.  The counts of these
        # different kinds of things are used for computing offsets for where to
        # store the times recorded during each batch.
        self.hook_names = hook_names
        self.scorer_name_lists = scorer_name_lists
        self.num_losses = len(scorer_name_lists)
        self.num_aux_scores = sum(map(len, scorer_name_lists)) - \
            len(scorer_name_lists)

        # Offsets of different events that we track while running a batch.
        #
        # The events at the beginning and end of train/test_on_batch() are the
        # same, so we can calculate their offsets here in the base class, given
        # the number of middle step time values to skip (new offset returned by
        # init_middle()).
        #
        # The offsets are indices into the list of times for that run of
        # train/test_on_batch().
        #
        # Lengths of each section:
        # - Start: 1.
        # - On begin: 1 + 2 * num hooks + 1.
        # - Forward: 1 + 1.
        # - Losses: 1 + 2 * num losses + 1.
        # - Aux scorers: 1 + 2 * num auxiliary scorers + 1.
        #   (train_on_batch() does additional work here...)
        # - On end: 1 + 2 * num hooks + 1.
        # - Stop: 1.
        self.start_offset = 0
        self.on_begin_offset = self.start_offset + 1
        self.forward_offset = \
            self.on_begin_offset + 1 + 2 * len(self.hook_names) + 1
        self.losses_offset = self.forward_offset + 2
        self.aux_scores_offset = \
            self.losses_offset + 1 + 2 * self.num_losses + 1
        middle_offset = self.aux_scores_offset + 1 + 2 * self.num_aux_scores + 1
        self.on_end_offset = self.init_middle(middle_offset)
        self.stop_offset = \
            self.on_end_offset + 1 + 2 * len(self.hook_names) + 1

        # The number of times we record the time per batch (row width).
        self.marks_per_batch = self.stop_offset + 1

        # The matrix of batch x index.
        shape = self.cache_size, self.marks_per_batch
        self.marks = np.zeros(shape, 'float64')

        # Where we are in the matrix during execution.
        self.cache_slot_id = None
        self.mark_id = None

    def start(self):
        if self.cache_used < self.cache_size:
            self.cache_slot_id = self.cache_used
            self.cache_used += 1
        else:
            self.cache_slot_id = np.random.randint(self.cache_size)
        self.mark_id = 0
        self.mark()

    def mark(self):
        self.marks[self.cache_slot_id, self.mark_id] = time()
        self.mark_id += 1

    def stop(self):
        self.mark()
        assert self.mark_id == self.marks_per_batch

    def ns_int_summary_stats(cls, ff, num_quantiles):
        ff_sorted = sorted(ff)
        quantiles = []
        for i in range(num_quantiles + 1):
            x = i * len(ff_sorted) // num_quantiles
            if x == len(ff_sorted):
                x -= 1
            f = ff_sorted[x]
            quantiles.append(f)
        return {
            'count': len(ff),
            'sum': int(ff.sum()),
            'mean': int(ff.mean()),
            'std': int(ff.std()),
            'min': int(ff.min()),
            'max': int(ff.max()),
            'quantiles': list(map(int, quantiles)),
        }

    def duration_stats(self, times, start_column, stop_column, num_quantiles):
        durations = times[:self.cache_used, stop_column] - \
            times[:self.cache_used, start_column]
        return self.ns_int_summary_stats(durations, num_quantiles)

    def middle_steps_stats(self, tt, num_quantiles):
        raise NotImplementedError

    def stats(self, num_quantiles):
        # Get times as nanoseconds since the start of the training batch.
        tt = self.marks - np.expand_dims(self.marks[:, 0], 1)
        tt *= 1e9

        # "On begin" hook callbacks.
        start = self.on_begin_offset
        stop = self.on_begin_offset + 1 + len(self.hook_names) * 2
        t_on_begin = self.duration_stats(tt, start, stop, num_quantiles)

        tt_on_begin = []
        for i in range(len(self.hook_names)):
            start = self.on_begin_offset + 1 + i * 2
            stop = self.on_begin_offset + 1 + i * 2 + 1
            t_on_begin = self.duration_stats(tt, start, stop, num_quantiles)
            tt_on_begin.append(t_on_begin)

        # Forward.
        start = self.forward_offset
        stop = self.forward_offset + 1
        t_forward = self.duration_stats(tt, start, stop, num_quantiles)

        # Losses.
        start = self.losses_offset
        stop = self.losses_offset + 1 + self.num_losses * 2
        t_loss = self.duration_stats(tt, start, stop, num_quantiles)

        tt_loss = []
        for i in range(self.num_losses):
            start = self.losses_offset + 1 + i * 2
            stop = self.losses_offset + 1 + i * 2 + 1
            t_loss = self.duration_stats(tt, start, stop, num_quantiles)
            tt_loss.append(t_loss)

        # Auxiliary scores.
        start = self.aux_scores_offset
        stop = self.aux_scores_offset + 1 * self.num_aux_scores * 2
        t_aux_score = self.duration_stats(tt, start, stop, num_quantiles)

        tt_aux_score = []
        for i in range(self.num_aux_scores):
            start = self.aux_scores_offset + 1 + i * 2
            stop = self.aux_scores_offset + 1 + i * 2 + 1
            t_aux_score = self.duration_stats(tt, start, stop, num_quantiles)
            tt_aux_score.append(t_aux_score)

        # Handle the middle steps that are different between train and test
        # batches.
        middle_name2stats = self.middle_steps_stats(tt, num_quantiles)

        # "On end" hook callbacks.
        start = self.on_end_offset
        stop = self.on_end_offset + 1
        t_on_end = self.duration_stats(tt, start, stop, num_quantiles)

        tt_on_end = []
        for i in range(len(self.hook_names)):
            start = self.on_end_offset + 1 + i * 2
            stop = self.on_end_offset + 1 + i * 2 + 1
            t_on_end = self.duration_stats(tt, start, stop, num_quantiles)
            tt_on_end.append(t_on_end)

        # End-to-end times.
        start = self.start_offset
        stop = self.stop_offset
        t_all = self.duration_stats(tt, start, stop, num_quantiles)

        # Gather the loss/aux scorer timings into a scorer timing stats list of
        # lists corresponding to the scorer name lists.
        ttt_score = []
        aux_score_index = 0
        for y_index, scorer_names in enumerate(self.scorer_name_lists):
            tt_score = [tt_loss[y_index]]
            for name in scorer_names[1:]:
                tt_score.append(tt_aux_score[aux_score_index])
                aux_score_index += 1
            ttt_score.append(tt_score)
        assert aux_score_index == len(tt_aux_score)

        # Gather the computed stats into a dict, including the stats for the
        # custom middle steps.
        ret = {
            'all': t_all,

            'forward': t_forward,

            'scorer_names': self.scorer_name_lists,
            'scorer_loss': t_loss,
            'scorer_aux': t_aux_score,
            'scorer_each': ttt_score,

            'hook_names': self.hook_names,
            'hook_on_begin': t_on_begin,
            'hook_on_begin_each': tt_on_begin,
            'hook_on_end': t_on_end,
            'hook_on_end_each': tt_on_end,
        }
        for name in middle_name2stats:
            assert name not in ret
        ret.update(middle_name2stats)
        return ret


class TrainOnBatchTimer(SplitOnBatchTimer):
    """
    Model.train_on_batch() timing statistics.

    See SplitOnBatchTimer for details.
    """

    def init_middle(self, middle_offset):
        self.backward_offset = middle_offset
        self.optim_offset = self.backward_offset + 2
        return self.optim_offset + 2

    def middle_steps_stats(self, tt, num_quantiles):
        # Backward.
        start = self.backward_offset
        stop = self.backward_offset + 1
        t_backward = self.duration_stats(tt, start, stop, num_quantiles)

        # Optimizer.
        start = self.optim_offset
        stop = self.optim_offset + 1
        t_optim = self.duration_stats(tt, start, stop, num_quantiles)

        return {
            'backward': t_backward,
            'optim': t_optim,
        }


class TestOnBatchTimer(SplitOnBatchTimer):
    """
    Model.test_on_batch() timing statistics.

    See SplitOnBatchTimer for details.
    """

    def init_middle(self, middle_offset):
        return middle_offset

    def middle_steps_stats(self, tt, num_quantiles):
        return {}


class BatchTimer(object):
    """
    Efficient fine-grained intra-batch timing statistics during Model training.

    Contains two collections of timing statistics: train and test.
    """

    def __init__(self, cache_size, hook_names, scorer_name_lists):
        self.train = TrainOnBatchTimer(
            cache_size, hook_names, scorer_name_lists)
        self.test = TestOnBatchTimer(cache_size, hook_names, scorer_name_lists)