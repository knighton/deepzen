import numpy as np

from ... import api as Z
from ...data.dataset import Dataset
from ...data.ram_split import RamSplit
from ...data.split import Split
from ...data import unpack_dataset
from ...spy import unpack_spies
from ...optim import unpack_optimizer
from ...meter import unpack_meter_lists
from ...util.py import require_kwargs_after
from .batch_timer import BatchTimer
from .training_cursor import TrainingCursor
from .trainer import Trainer


class Model(object):
    def __init__(self):
        self._is_built = False

    def is_built(self):
        return self._is_built

    def build(self):
        raise NotImplementedError

    def ensure_built(self):
        if self._is_built:
            return
        self.build()
        self._is_built = True

    def params(self):
        raise NotImplementedError

    def forward(self, xx, is_training):
        raise NotImplementedError

    def train_on_batch(self, trainer, xx, yy_true):
        # Start timing the whole method.
        t = trainer.batch_timer.train
        t.start()

        # 1. Execute "on begin" callbacks.
        t.mark()
        for spy in trainer.spies:
            t.mark()
            spy.on_train_on_batch_begin()
            t.mark()
        t.mark()

        losses = []
        with Z.autograd_record():
            # 2. Forward propagate.
            t.mark()
            yy_pred = self.forward(xx, True)
            t.mark()

            # 3. Compute the loss of each output.
            t.mark()
            for meters, y_true, y_pred in \
                    zip(trainer.meter_lists, yy_true, yy_pred):
                get_loss = meters[0]
                t.mark()
                loss = Z.mean(get_loss(y_true, y_pred))
                t.mark()
                losses.append(loss)
            t.mark()

        # 4. Compute any additional metrics of each output.
        metric_lists = []
        t.mark()
        for i, (meters, y_true, y_pred) in \
                enumerate(zip(trainer.meter_lists, yy_true, yy_pred)):
            loss = Z.scalar(losses[i])
            metrics = [loss]
            for extra_meter in meters[1:]:
                t.mark()
                metric = Z.mean(extra_meter(y_true, y_pred))
                t.mark()
                metric = Z.scalar(metric)
                metrics.append(metric)
            metric_lists.append(metrics)
        t.mark()

        # 5. Backpropagate gradients.
        grads = [Z.ones((1,), 'float32') for x in losses]
        t.mark()
        Z.backward(losses, grads)
        t.mark()

        # 6. Perform one step of the optimizer.
        t.mark()
        trainer.optimizer.step()
        t.mark()

        # 7. Execute "on end" callbacks.
        t.mark()
        for spy in trainer.spies:
            t.mark()
            spy.on_train_on_batch_end()
            t.mark()
        t.mark()

        # Stop timing the whole method.
        t.stop()

        return metric_lists

    def test_on_batch(self, trainer, xx, yy_true):
        # Start timing the whole method.
        t = trainer.batch_timer.test
        t.start()

        # 1. Execute "on begin" callbacks.
        t.mark()
        for spy in trainer.spies:
            t.mark()
            spy.on_test_on_batch_begin()
            t.mark()
        t.mark()

        # 2. Forward propagate.
        t.mark()
        yy_pred = self.forward(xx, False)
        t.mark()

        # 3. Compute the loss of each output.
        losses = []
        t.mark()
        for i, (meters, y_true, y_pred) in \
                enumerate(zip(trainer.meter_lists, yy_true, yy_pred)):
            get_loss = meters[0]
            t.mark()
            loss = Z.mean(get_loss(y_true, y_pred))
            t.mark()
            losses.append(loss)
        t.mark()

        # 3. Compute any additional metrics of each output.  (This could be done
        #    in the same loop as computing losses, but is done separately so
        #    timings can be compared directly.)
        metric_lists = []
        t.mark()
        for i, (meters, y_true, y_pred) in \
                enumerate(zip(trainer.meter_lists, yy_true, yy_pred)):
            loss = Z.scalar(losses[i])
            metrics = [loss]
            for extra_meter in meters[1:]:
                t.mark()
                metric = Z.mean(extra_meter(y_true, y_pred))
                t.mark()
                metric = Z.scalar(metric)
                metrics.append(metric)
            metric_lists.append(metrics)
        t.mark()

        # 4. Execute "on end" callbacks.
        t.mark()
        for spy in trainer.spies:
            t.mark()
            spy.on_test_on_batch_end()
            t.mark()
        t.mark()

        # Stop timing the whole method.
        t.stop()

        return metric_lists

    def _fit_on_batch(self, trainer, cursor, is_training, xx, yy,
                      train_metric_lists, test_metric_lists):
        xx = [Z.constant(x) for x in xx]
        yy = [Z.constant(y) for y in yy]

        if is_training:
            batch_metric_lists = self.train_on_batch(trainer, xx, yy)
            split_metric_lists = train_metric_lists
        else:
            batch_metric_lists = self.test_on_batch(trainer, xx, yy)
            split_metric_lists = test_metric_lists

        for i, batch_metrics in enumerate(batch_metric_lists):
            for j, batch_metric in enumerate(batch_metrics):
                split_metric_lists[i][j].append(batch_metric)

        cursor.completed_batch(is_training)

    def _fit_epoch(self, dataset, trainer, cursor):
        for spy in trainer.spies:
            spy.on_epoch_begin(cursor.epoch, cursor.batches_per_epoch)

        train_metric_lists = []
        test_metric_lists = []
        for meters in trainer.meter_lists:
            train_metric_lists.append([[] for x in meters])
            test_metric_lists.append([[] for x in meters])

        for (xx, yy), is_training in dataset.each_batch(cursor.batch_size):
            self._fit_on_batch(trainer, cursor, is_training, xx, yy,
                               train_metric_lists, test_metric_lists)

        for split_metric_lists in [train_metric_lists, test_metric_lists]:
            for i, column in enumerate(split_metric_lists):
                for j, values in enumerate(column):
                    split_metric_lists[i][j] = float(np.mean(values))

        for spy in trainer.spies:
            spy.on_epoch_end(train_metric_lists, test_metric_lists)

        return train_metric_lists, test_metric_lists

    def resume_fit(self, dataset, trainer, cursor):
        for spy in trainer.spies:
            spy.on_fit_begin(trainer.batch_timer.meter_name_lists,
                             cursor.begin_epoch, cursor.end_epoch)

        for epoch in range(cursor.begin_epoch, cursor.end_epoch):
            train_metric_lists, test_metric_lists = \
                self._fit_epoch(dataset, trainer, cursor)

        for spy in trainer.spies:
            spy.on_fit_end()

        return dataset, trainer, cursor

    @require_kwargs_after(3)
    def fit(self, data, loss, test_frac=None, optim='adam', batch=64, start=0,
            stop=20, spy=None, timer_cache=10000):
        dataset = unpack_dataset(data, test_frac)
        y_sample_shapes = dataset.shapes()[1]
        meter_lists = unpack_meter_lists(loss, y_sample_shapes)
        optimizer = unpack_optimizer(optim)
        spies = unpack_spies(spy)
        assert isinstance(batch, int)
        assert 0 < batch
        batch_size = batch
        assert isinstance(start, int)
        assert isinstance(stop, int)
        assert 0 <= start <= stop
        begin_epoch = start
        end_epoch = stop
        assert isinstance(timer_cache, int)
        assert 0 < timer_cache
        timer_cache_size = timer_cache
        batch_timer = BatchTimer.init_getting_names(
            timer_cache_size, spies, meter_lists)

        cursor = TrainingCursor.start_from_dataset(
            dataset, begin_epoch, end_epoch, batch_size)
        trainer = Trainer(meter_lists, optimizer, spies, batch_timer)

        self.ensure_built()
        trainer.optimizer.set_params(self.params())

        return self.resume_fit(dataset, trainer, cursor)

    @require_kwargs_after(2)
    def fit_reg(self, data, test_frac=None, optim='adam', batch=64, start=0,
                stop=20, spy=None, timer_cache=10000):
        loss = [['mean_squared_error']]
        return self.fit(data, loss, test_frac=test_frac, optim=optim,
                        batch=batch, start=start, stop=stop, spy=spy,
                        timer_cache=timer_cache)

    @require_kwargs_after(2)
    def fit_clf(self, data, test_frac=None, optim='adam', batch=64, start=0,
                stop=20, spy=None, timer_cache=10000):
        loss = [['cross_entropy', 'accuracy']]
        return self.fit(data, loss, test_frac=test_frac, optim=optim,
                        batch=batch, start=start, stop=stop, spy=spy,
                        timer_cache=timer_cache)
