from .base.callback import Callback


class RowPerEpoch(Callback):
    def __init__(self, epoch_cols=10, crit_cols=9, crit_decimal_places=4):
        self.epoch_cols = epoch_cols
        assert crit_decimal_places + 2 <= crit_cols
        self.crit_cols = crit_cols
        self.crit_decimal_places = crit_decimal_places

    def on_fit_begin(self, crit_name_lists, epoch_offset, epochs):
        split_cols = 0
        for crit_names in crit_name_lists:
            count = len(crit_names)
            split_cols += count * self.crit_cols + (count - 1)

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

    def draw_output(self, crits):
        ss = []
        for crit in crits:
            fmt = '%%%d.%df' % (self.crit_cols, self.crit_decimal_places)
            ss.append(fmt % crit)
        return ' '.join(ss)

    def draw_split(self, crit_lists):
        ss = []
        for crits in crit_lists:
            ss.append(self.draw_output(crits))
        return ' : '.join(ss)

    def on_epoch_end(self, train_crit_lists, test_crit_lists):
        train_text = self.draw_split(train_crit_lists)
        test_text = self.draw_split(test_crit_lists)
        fmt = '    | %%%dd | %%s | %%s |' % self.epoch_cols
        print(fmt % (self.epoch, train_text, test_text))

    def on_fit_end(self):
        print(self.horizontal_bar)
