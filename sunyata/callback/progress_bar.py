from tqdm import tqdm

from .base.callback import Callback


class ProgressBar(Callback):
    def on_epoch_begin(self, num_batches):
        self.bar = tqdm(total=num_batches, leave=False)

    def on_train_on_batch_begin(self):
        self.bar.update(1)

    def on_test_on_batch_begin(self):
        self.bar.update(1)

    def on_epoch_end(self):
        self.bar.close()
