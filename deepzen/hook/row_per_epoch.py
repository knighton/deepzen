from .base.hook import Hook


class RowPerEpoch(Hook):
    def __init__(self, epoch_cols=10, metric_cols=9, metric_decimal_places=4):
        self.epoch_cols = epoch_cols
        assert metric_decimal_places + 2 <= metric_cols
        self.metric_cols = metric_cols
        self.metric_decimal_places = metric_decimal_places

    def on_fit_begin(self, metric_name_lists, epoch_offset, epochs):
        split_cols = 0
        for metric_names in metric_name_lists:
            count = len(metric_names)
            split_cols += count * self.metric_cols + (count - 1)

        self.horizontal_bar = '    +-%s-+-%s-+-%s-+' % \
            ('-' * self.epoch_cols, '-' * split_cols, '-' * split_cols)

        from_to_str = '%d->%d' % (epoch_offset, epoch_offset + epochs)
        print(self.horizontal_bar)

        fmt = '    | %%%ds | %%%ds | %%%ds |' % \
            (self.epoch_cols, split_cols, split_cols)
        print(fmt % ('Epochs', 'Train', 'Test'))
        fmt = '    | %%%ds | %%s | %%s |' % self.epoch_cols
        print(fmt % (from_to_str, ' ' * split_cols, ' ' * split_cols))
        print(self.horizontal_bar)

    def on_epoch_begin(self, epoch, num_batches):
        self.epoch = epoch

    def draw_output(self, metrics):
        ss = []
        for metric in metrics:
            fmt = '%%%d.%df' % (self.metric_cols, self.metric_decimal_places)
            ss.append(fmt % metric)
        return ' '.join(ss)

    def draw_split(self, metric_lists):
        ss = []
        for metrics in metric_lists:
            ss.append(self.draw_output(metrics))
        return ' : '.join(ss)

    def on_epoch_end(self, train_metric_lists, test_metric_lists):
        train_text = self.draw_split(train_metric_lists)
        test_text = self.draw_split(test_metric_lists)
        fmt = '    | %%%dd | %%s | %%s |' % self.epoch_cols
        print(fmt % (self.epoch, train_text, test_text))

    def on_fit_end(self):
        print(self.horizontal_bar)
