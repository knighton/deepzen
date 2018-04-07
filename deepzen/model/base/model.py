import numpy as np

from ... import api as Z
from ...data.dataset import Dataset
from ...data.ram_split import RamSplit
from ...data.split import Split
from ...spy import unpack_spies
from ...optim import unpack_optimizer
from ...meter import unpack_meter_lists
from ...util.py import require_kwargs_after
from .batch_timer import BatchTimer


class Model(object):
    @classmethod
    def _unpack_split(cls, split):
        if isinstance(split, Split):
            return split

        xx, yy = split
        return RamSplit(xx, yy)

    @classmethod
    def _unpack_dataset(cls, dataset, test_frac=None):
        if isinstance(dataset, Dataset):
            assert test_frac is None
            return dataset

        if test_frac is not None:
            assert False, 'TODO: Perform train/test split.'

        train, test = dataset
        train = cls._unpack_split(train)
        test = cls._unpack_split(test)
        return Dataset(train, test)

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

    def train_on_batch(self, xx, yy_true, meter_lists, optim, spies, t):
        # Start timing the whole method.
        t.start()

        # 1. Execute "on begin" callbacks.
        t.mark()
        for spy in spies:
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
            for meters, y_true, y_pred in zip(meter_lists, yy_true, yy_pred):
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
                enumerate(zip(meter_lists, yy_true, yy_pred)):
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
        optim.step()
        t.mark()

        # 7. Execute "on end" callbacks.
        t.mark()
        for spy in spies:
            t.mark()
            spy.on_train_on_batch_end()
            t.mark()
        t.mark()

        # Stop timing the whole method.
        t.stop()

        return metric_lists

    def test_on_batch(self, xx, yy_true, meter_lists, spies, t):
        # Start timing the whole method.
        t.start()

        # 1. Execute "on begin" callbacks.
        t.mark()
        for spy in spies:
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
                enumerate(zip(meter_lists, yy_true, yy_pred)):
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
                enumerate(zip(meter_lists, yy_true, yy_pred)):
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
        for spy in spies:
            t.mark()
            spy.on_test_on_batch_end()
            t.mark()
        t.mark()

        # Stop timing the whole method.
        t.stop()

        return metric_lists

    def _fit_epoch(self, meter_lists, dataset, optim, batch_size, spies, timer,
                   epoch):
        for spy in spies:
            spy.on_epoch_begin(epoch, dataset.num_batches(batch_size))

        train_metric_lists = []
        test_metric_lists = []
        for meters in meter_lists:
            train_metric_lists.append([[] for x in meters])
            test_metric_lists.append([[] for x in meters])

        for (xx, yy), is_training in dataset.each_batch(batch_size):
            xx = [Z.constant(x) for x in xx]
            yy = [Z.constant(y) for y in yy]
            if is_training:
                batch_metric_lists = self.train_on_batch(
                    xx, yy, meter_lists, optim, spies, timer.train)
                split_metric_lists = train_metric_lists
            else:
                batch_metric_lists = self.test_on_batch(
                    xx, yy, meter_lists, spies, timer.test)
                split_metric_lists = test_metric_lists
            for i, batch_metrics in enumerate(batch_metric_lists):
                for j, batch_metric in enumerate(batch_metrics):
                    split_metric_lists[i][j].append(batch_metric)

        for split_metric_lists in [train_metric_lists, test_metric_lists]:
            for i, column in enumerate(split_metric_lists):
                for j, values in enumerate(column):
                    split_metric_lists[i][j] = float(np.mean(values))

        for spy in spies:
            spy.on_epoch_end(train_metric_lists, test_metric_lists)

        return train_metric_lists, test_metric_lists

    @require_kwargs_after(3)
    def fit(self, data, loss, test_frac=None, optim='sgd', batch=64,
            epoch_offset=0, epochs=20, spies=None, timer_cache=10000):
        dataset = self._unpack_dataset(data, test_frac)
        y_sample_shapes = dataset.shapes()[1]
        meter_lists = unpack_meter_lists(loss, y_sample_shapes)
        optim = unpack_optimizer(optim)
        spies = unpack_spies(spies)
        assert isinstance(batch, int)
        assert 0 < batch
        batch_size = batch
        assert isinstance(epoch_offset, int)
        assert 0 <= epoch_offset
        assert isinstance(epochs, int)
        assert 0 <= epochs
        assert isinstance(timer_cache, int)
        assert 0 < timer_cache
        timer_cache_size = timer_cache

        spy_names = [x.__class__.__name__ for x in spies]
        meter_name_lists = []
        for meters in meter_lists:
            meter_names = [x.__class__.__name__ for x in meters]
            meter_name_lists.append(meter_names)
        timer = BatchTimer(timer_cache_size, spy_names, meter_name_lists)

        self.ensure_built()
        optim.set_params(self.params())

        for spy in spies:
            spy.on_fit_begin(meter_name_lists, epoch_offset, epochs)

        for epoch in range(epoch_offset, epoch_offset + epochs):
            train_metric_lists, test_metric_lists = \
                self._fit_epoch(meter_lists, dataset, optim, batch_size, spies,
                                timer, epoch)

        for spy in spies:
            spy.on_fit_end()

    @require_kwargs_after(2)
    def fit_reg(self, data, test_frac=None, optim='sgd', batch=64,
                epoch_offset=0, epochs=20, spies=None, timer_cache=10000):
        loss = [['mean_squared_error']]
        return self.fit(data, loss, test_frac=test_frac, optim=optim,
                        batch=batch, epoch_offset=epoch_offset, epochs=epochs,
                        spies=spies, timer_cache=timer_cache)

    @require_kwargs_after(2)
    def fit_clf(self, data, test_frac=None, optim='sgd', batch=64,
                epoch_offset=0, epochs=20, spies=None, timer_cache=10000):
        loss = [['cross_entropy', 'accuracy']]
        return self.fit(data, loss, test_frac=test_frac, optim=optim,
                        batch=batch, epoch_offset=epoch_offset, epochs=epochs,
                        spies=spies, timer_cache=timer_cache)
