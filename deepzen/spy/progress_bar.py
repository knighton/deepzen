from tqdm import tqdm

from .base.registry import register_spy
from .base.spy import Spy


@register_spy
class ProgressBar(Spy):
    name = 'progress_bar'

    def on_epoch_begin(self, epoch, num_batches):
        self.bar = tqdm(total=num_batches, leave=False)

    def on_train_on_batch_begin(self):
        self.bar.update(1)

    def on_test_on_batch_begin(self):
        self.bar.update(1)

    def on_epoch_end(self, train_meter_lists, test_meter_lists):
        self.bar.close()
