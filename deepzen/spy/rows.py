from .base.registry import register_spy
from .base.spy import Spy


@register_spy
class Rows(Spy):
    name = 'rows'

    def __init__(self, epoch_cols=10, score_cols=9, score_decimal_places=4):
        self.epoch_cols = epoch_cols
        assert score_decimal_places + 2 <= score_cols
        self.score_cols = score_cols
        self.score_decimal_places = score_decimal_places

    def on_fit_begin(self, scorer_name_lists, epoch_offset, epochs):
        split_cols = 0
        for scorer_names in scorer_name_lists:
            count = len(scorer_names)
            split_cols += count * self.score_cols + (count - 1)

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

    def draw_output(self, scores):
        ss = []
        for score in scores:
            fmt = '%%%d.%df' % (self.score_cols, self.score_decimal_places)
            ss.append(fmt % score)
        return ' '.join(ss)

    def draw_split(self, score_lists):
        ss = []
        for scores in score_lists:
            ss.append(self.draw_output(scores))
        return ' : '.join(ss)

    def on_epoch_end(self, train_score_lists, test_score_lists):
        train_text = self.draw_split(train_score_lists)
        test_text = self.draw_split(test_score_lists)
        fmt = '    | %%%dd | %%s | %%s |' % self.epoch_cols
        print(fmt % (self.epoch, train_text, test_text))

    def on_fit_end(self):
        print(self.horizontal_bar)
