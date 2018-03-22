import numpy as np

from .. import api as Z
from ..io.dataset import Dataset
from ..io.ram_split import RamSplit
from ..io.split import Split
from ..metric.loss import unpack_loss
from ..metric import unpack_metric
from ..optim import get_optimizer
from ..util.py import require_kwargs_after
from ..view import unpack_views
from .batch_timer import TestOnBatchTimer, TrainOnBatchTimer


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

    @classmethod
    def _parse_metric_lists_str(cls, s):
        ss = s.split(' ')
        return [s.split(',') for s in ss]

    @classmethod
    def _unpack_loss_and_metrics(cls, arg, y_sample_shape):
        if not isinstance(arg, (list, tuple)):
            arg = [arg]
        metrics = []
        metrics.append(unpack_loss(arg[0], y_sample_shape))
        for item in arg[1:]:
            metrics.append(unpack_metric(item, y_sample_shape))
        return metrics

    @classmethod
    def _unpack_metric_lists(cls, arg, y_sample_shapes):
        if isinstance(arg, str):
            if ' ' in arg or ',' in arg:
                arg = cls._parse_metric_lists_str(arg)
            else:
                arg = [arg]
        metric_lists = []
        assert len(arg) == len(y_sample_shapes)
        for item, y_sample_shape in zip(arg, y_sample_shapes):
            loss_and_metrics = \
                cls._unpack_loss_and_metrics(item, y_sample_shape)
            metric_lists.append(loss_and_metrics)
        return metric_lists

    def __init__(self, spec):
        self.spec = spec
        self.layer = spec.build()

    def forward(self, xx, is_training):
        x, = xx
        y_pred = self.layer.forward(x, is_training)
        return [y_pred]

    def train_on_batch(self, xx, yy_true, compute_metric_lists, optim, views,
                       t):
        # Start timing the whole method.
        t.start()

        # 1. Execute "on begin" views.
        t.mark()
        for view in views:
            t.mark()
            view.on_train_on_batch_begin()
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
            for compute_metrics, y_true, y_pred in \
                    zip(compute_metric_lists, yy_true, yy_pred):
                compute_loss = compute_metrics[0]
                t.mark()
                loss = Z.mean(compute_loss(y_true, y_pred))
                t.mark()
                losses.append(loss)
            t.mark()

        # 4. Compute any additional metrics of each output.
        metric_lists = []
        t.mark()
        for i, (compute_metrics, y_true, y_pred) in \
                enumerate(zip(compute_metric_lists, yy_true, yy_pred)):
            loss = Z.scalar(losses[i])
            metrics = [loss]
            for compute_metric in compute_metrics[1:]:
                t.mark()
                metric = Z.mean(compute_metric(y_true, y_pred))
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

        # 7. Execute "on end" views.
        t.mark()
        for view in views:
            t.mark()
            view.on_train_on_batch_end()
            t.mark()
        t.mark()

        # Stop timing the whole method.
        t.stop()

        return metric_lists

    def test_on_batch(self, xx, yy_true, compute_metric_lists, views, t):
        # Start timing the whole method.
        t.start()

        # 1. Execute "before" views.
        t.mark()
        for view in views:
            t.mark()
            view.on_test_on_batch_begin()
            t.mark()
        t.mark()

        # 2. Forward propagate.
        t.mark()
        yy_pred = self.forward(xx, False)
        t.mark()

        # 3. Compute the loss of each output.
        losses = []
        t.mark()
        for i, (compute_metrics, y_true, y_pred) in \
                enumerate(zip(compute_metric_lists, yy_true, yy_pred)):
            compute_loss = compute_metrics[0]
            t.mark()
            loss = Z.mean(compute_loss(y_true, y_pred))
            t.mark()
            losses.append(loss)
        t.mark()

        # 3. Compute any additional metrics of each output.  (This could be
        #    done in the same loop as computing losses, but is done separately
        #    so timings can be compared directly.)
        metric_lists = []
        t.mark()
        for i, (compute_metrics, y_true, y_pred) in \
                enumerate(zip(compute_metric_lists, yy_true, yy_pred)):
            loss = Z.scalar(losses[i])
            metrics = [loss]
            for compute_metric in compute_metrics[1:]:
                t.mark()
                metric = Z.mean(compute_metric(y_true, y_pred))
                t.mark()
                metric = Z.scalar(metric)
                metrics.append(metric)
            metric_lists.append(metrics)
        t.mark()

        # 4. Execute "after" views.
        t.mark()
        for view in views:
            t.mark()
            view.on_test_on_batch_end()
            t.mark()
        t.mark()

        # Stop timing the whole method.
        t.stop()

        return metric_lists

    def _fit_epoch(self, compute_metric_lists, dataset, optim, batch_size,
                   views, train_timer, test_timer, epoch):
        for view in views:
            view.on_epoch_begin(epoch, dataset.num_batches(batch_size))

        train_metric_lists = []
        test_metric_lists = []
        for compute_metrics in compute_metric_lists:
            train_metric_lists.append([[] for x in compute_metrics])
            test_metric_lists.append([[] for x in compute_metrics])

        for (xx, yy), is_training in dataset.each_batch(batch_size):
            xx = [Z.constant(x) for x in xx]
            yy = [Z.constant(y) for y in yy]
            if is_training:
                batch_metric_lists = self.train_on_batch(
                    xx, yy, compute_metric_lists, optim, views, train_timer)
                split_metric_lists = train_metric_lists
            else:
                batch_metric_lists = self.test_on_batch(
                    xx, yy, compute_metric_lists, views, test_timer)
                split_metric_lists = test_metric_lists
            for i, batch_metrics in enumerate(batch_metric_lists):
                for j, batch_metric in enumerate(batch_metrics):
                    split_metric_lists[i][j].append(batch_metric)

        for split_metric_lists in [train_metric_lists, test_metric_lists]:
            for i, column in enumerate(split_metric_lists):
                for j, values in enumerate(column):
                    split_metric_lists[i][j] = float(np.mean(values))

        for view in views:
            view.on_epoch_end(train_metric_lists, test_metric_lists)

        return train_metric_lists, test_metric_lists

    @require_kwargs_after(3)
    def fit(self, data, loss, test_frac=None, optim='sgd', batch=64,
            epoch_offset=0, epochs=20, view=None, timer_cache=10000):
        dataset = self._unpack_dataset(data, test_frac)
        y_sample_shapes = dataset.shapes()[0]
        compute_metric_lists = self._unpack_metric_lists(loss, y_sample_shapes)
        optim = get_optimizer(optim)
        views = unpack_views(view)
        assert isinstance(batch, int)
        assert 0 < batch
        batch_size = batch
        assert isinstance(epoch_offset, int)
        assert 0 <= epoch_offset
        assert isinstance(epochs, int)
        assert 0 <= epochs

        timer_cache_size = timer_cache
        view_names = [x.__class__.__name__ for x in views]
        metric_name_lists = []
        for compute_metrics in compute_metric_lists:
            metric_names = [x.__class__.__name__ for x in compute_metrics]
            metric_name_lists.append(metric_names)
        train_timer = TrainOnBatchTimer(
            timer_cache_size, view_names, metric_name_lists)
        test_timer = TestOnBatchTimer(
            timer_cache_size, view_names, metric_name_lists)

        optim.set_params(self.layer.params())

        for view in views:
            view.on_fit_begin(metric_name_lists, epoch_offset, epochs)

        for epoch in range(epoch_offset, epoch_offset + epochs):
            train_metric_lists, test_metric_lists = \
                self._fit_epoch(compute_metric_lists, dataset, optim,
                                batch_size, views, train_timer, test_timer,
                                epoch)

        for view in views:
            view.on_fit_end()

    @require_kwargs_after(2)
    def fit_reg(self, data, test_frac=None, optim='sgd', batch=64,
                epoch_offset=0, epochs=20, view=None, timer_cache=10000):
        loss = [['mean_squared_error']]
        return self.fit(data, loss, test_frac=test_frac, optim=optim,
                        batch=batch, epoch_offset=epoch_offset, epochs=epochs,
                        view=view, timer_cache=timer_cache)

    @require_kwargs_after(2)
    def fit_clf(self, data, test_frac=None, optim='sgd', batch=64,
                epoch_offset=0, epochs=20, view=None, timer_cache=10000):
        loss = [['cross_entropy', 'accuracy']]
        return self.fit(data, loss, test_frac=test_frac, optim=optim,
                        batch=batch, epoch_offset=epoch_offset, epochs=epochs,
                        view=view, timer_cache=timer_cache)
