import json
import numpy as np
from time import time


class TrainBatchTimer(object):
    def __init__(self, num_batches, callback_names, crit_name_lists):
        self.num_batches = num_batches
        self.callback_names = callback_names
        self.crit_name_lists = crit_name_lists
        self.num_losses = len(crit_name_lists)
        self.num_metrics = sum(map(len, crit_name_lists)) - len(crit_name_lists)

        # Clock times recorded per run of train_on_batch():
        # - Start: 1.
        # - On begin: 1 + 2 * num callbacks + 1.
        # - Forward: 1 + 1.
        # - Losses: 1 + 2 * num losses + 1.
        # - Metrics: 1 + 2 * num metrics + 1.
        # - Backward: 1 + 1.
        # - Optim: 1 + 1.
        # - On end: 1 + 2 * num callbacks + 1.
        # - Stop: 1.
        self.start_offset = 0
        self.on_begin_offset = self.start_offset + 1
        self.forward_offset = \
            self.on_begin_offset + 1 + 2 * len(self.callback_names) + 1
        self.losses_offset = self.forward_offset + 2
        self.metrics_offset = self.losses_offset + 1 + 2 * self.num_losses + 1
        self.backward_offset = \
            self.metrics_offset + 1 + 2 * self.num_metrics + 1
        self.optim_offset = self.backward_offset + 2
        self.on_end_offset = self.optim_offset + 2
        self.stop_offset = \
            self.on_end_offset + 1 + 2 * len(self.callback_names) + 1
        self.times_per_batch = self.stop_offset + 1

        # The matrix of batch x timing.
        self.times = np.zeros((num_batches, self.times_per_batch), 'float64')

        # Where we are in the matrix during execution.
        self.batch_id = 0
        self.time_id = 0

    def start_train_on_batch(self):
        assert 0 <= self.batch_id < self.num_batches
        assert self.time_id == 0
        self.mark()

    def mark(self):
        self.times[self.batch_id, self.time_id] = time()
        self.time_id += 1

    def stop_train_on_batch(self):
        self.mark()
        assert self.time_id == self.times_per_batch
        self.batch_id += 1
        self.time_id = 0

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
        durations = times[:self.batch_id, stop_column] - \
            times[:self.batch_id, start_column]
        return self.ns_int_summary_stats(durations, num_quantiles)

    def analyze_to_json(self, num_quantiles):
        # Get times as nanoseconds since the start of the training batch.
        tt = self.times - np.expand_dims(self.times[:, 0], 1)
        tt *= 1e9

        # 1. On begin callbacks.
        start = self.on_begin_offset
        stop = self.on_begin_offset + 1 + len(self.callback_names) * 2
        t_on_begin = self.duration_stats(tt, start, stop, num_quantiles)

        tt_on_begin = []
        for i in range(len(self.callback_names)):
            start = self.on_begin_offset + 1 + i * 2
            stop = self.on_begin_offset + 1 + i * 2 + 1
            per_callback = self.duration_stats(tt, start, stop, num_quantiles)
            tt_on_begin.append(per_callback)

        # 2. Forward.
        start = self.forward_offset
        stop = self.forward_offset + 1
        t_forward = self.duration_stats(tt, start, stop, num_quantiles)

        # 3. Losses.
        start = self.losses_offset
        stop = self.losses_offset + 1 + self.num_losses * 2
        t_loss = self.duration_stats(tt, start, stop, num_quantiles)

        tt_loss = []
        for i in range(self.num_losses):
            start = self.losses_offset + 1 + i * 2
            stop = self.losses_offset + 1 + i * 2 + 1
            per_y = self.duration_stats(tt, start, stop, num_quantiles)
            tt_loss.append(per_y)

        # 4. Metrics.
        start = self.metrics_offset
        stop = self.metrics_offset + 1 * self.num_metrics * 2
        t_metric = self.duration_stats(tt, start, stop, num_quantiles)

        tt_metric = []
        for i in range(self.num_metrics):
            start = self.metrics_offset + 1 + i * 2
            stop = self.metrics_offset + 1 + i * 2 + 1
            per_metric = self.duration_stats(tt, start, stop, num_quantiles)
            tt_metric.append(per_metric)

        # 5. Backward.
        start = self.backward_offset
        stop = self.backward_offset + 1
        t_backward = self.duration_stats(tt, start, stop, num_quantiles)

        # 6. Optimizer.
        start = self.optim_offset
        stop = self.optim_offset + 1
        t_optim = self.duration_stats(tt, start, stop, num_quantiles)

        # 7. On end callbacks.
        start = self.on_end_offset
        stop = self.on_end_offset + 1
        t_on_end = self.duration_stats(tt, start, stop, num_quantiles)

        tt_on_end = []
        for i in range(len(self.callback_names)):
            start = self.on_end_offset + 1 + i * 2
            stop = self.on_end_offset + 1 + i * 2 + 1
            per_callback = self.duration_stats(tt, start, stop, num_quantiles)
            tt_on_end.append(per_callback)

        # End-to-end.
        start = self.start_offset
        stop = self.stop_offset
        t_all = self.duration_stats(tt, start, stop, num_quantiles)

        # Crit lists.
        ttt_crit = []
        metric_index = 0
        for y_index, crit_names in enumerate(self.crit_name_lists):
            tt_crit = [tt_loss[y_index]]
            for name in crit_names[1:]:
                tt_crit.append(tt_metric[metric_index])
                metric_index += 1
            ttt_crit.append(tt_crit)
        assert metric_index == len(tt_metric)

        # Gather into a dict.
        names = {
            'callbacks': self.callback_names,
            'crit': self.crit_name_lists,
        }
        duration_stats = {
            'all': t_all,
            'on_begin': t_on_begin,
            'on_begin_each': tt_on_begin,
            'forward': t_forward,
            'loss': t_loss,
            'metric': t_metric,
            'crit_each': ttt_crit,
            'backward': t_backward,
            'optim': t_optim,
            'on_end': t_on_end,
            'on_end_each': tt_on_end,
        }
        return {
            'name': names,
            'time': duration_stats,
        }

    def summary(self, num_quantiles=100):
        x = self.analyze_to_json(num_quantiles)
        print(json.dumps(x, indent=4, sort_keys=True))
