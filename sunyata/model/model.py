import json
import numpy as np
from time import time

from .. import api as Z
from ..callback import unpack_callbacks
from ..crit.loss import unpack_loss
from ..crit.metric import unpack_metric
from ..iter.dataset import Dataset
from ..iter.ram_split import RamSplit
from ..iter.split import Split
from ..optim import unpack_optim
from ..util.py import require_kwargs_after
from .train_batch_timer import TrainBatchTimer


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
    def _parse_crit_lists_str(cls, s):
        ss = s.split(' ')
        return [s.split(',') for s in ss]

    @classmethod
    def _unpack_loss_and_metrics(cls, arg, y_sample_shape):
        if not isinstance(arg, (list, tuple)):
            arg = [arg]
        crits = []
        crits.append(unpack_loss(arg[0], y_sample_shape))
        for item in arg[1:]:
            crits.append(unpack_metric(item, y_sample_shape))
        return crits

    @classmethod
    def _unpack_crit_lists(cls, arg, y_sample_shapes):
        if isinstance(arg, str):
            if ' ' in arg or ',' in arg:
                arg = cls._parse_crit_lists_str(arg)
            else:
                arg = [arg]
        crit_lists = []
        assert len(arg) == len(y_sample_shapes)
        for item, y_sample_shape in zip(arg, y_sample_shapes):
            loss_and_metrics = \
                cls._unpack_loss_and_metrics(item, y_sample_shape)
            crit_lists.append(loss_and_metrics)
        return crit_lists

    def __init__(self, spec):
        self.spec = spec
        self.layer, self.out_form = spec.build()

    def forward(self, xx, is_training):
        x, = xx
        y_pred = self.layer.forward(x)
        return [y_pred]

    def train_on_batch(self, xx, yy_true, compute_crit_lists, optim, callbacks,
                       t):
        # Start timing the whole method.
        t.start_train_on_batch()

        # 1. Execute "on begin" callbacks.
        t.mark()
        for callback in callbacks:
            t.mark()
            callback.on_train_on_batch_begin()
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
            for compute_crits, y_true, y_pred in \
                    zip(compute_crit_lists, yy_true, yy_pred):
                compute_loss = compute_crits[0]
                t.mark()
                loss = compute_loss(y_true, y_pred)
                t.mark()
                losses.append(loss)
            t.mark()

        # 4. Compute any additional metrics of each output.
        crit_lists = []
        t.mark()
        for i, (compute_crits, y_true, y_pred) in \
                enumerate(zip(compute_crit_lists, yy_true, yy_pred)):
            loss = Z.variable_to_numpy(losses[i])[0]
            crits = [loss]
            for compute_metric in compute_crits[1:]:
                t.mark()
                metric = compute_metric(y_true, y_pred)
                t.mark()
                metric = Z.variable_to_numpy(metric)[0]
                crits.append(metric)
            crit_lists.append(crits)
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
        for callback in callbacks:
            t.mark()
            callback.on_train_on_batch_end()
            t.mark()
        t.mark()

        # Stop timing the whole method.
        t.stop_train_on_batch()

        return crit_lists

    def test_on_batch(self, xx, yy_true, compute_crit_lists, callbacks):
        # 1. Execute "before" callbacks.
        for callback in callbacks:
            callback.on_test_on_batch_begin()

        # 2. Forward propagate.
        yy_pred = self.forward(xx, False)

        # 3. Compute the loss of each output.
        losses = []
        for i, (compute_crits, y_true, y_pred) in \
                enumerate(zip(compute_crit_lists, yy_true, yy_pred)):
            compute_loss = compute_crits[0]
            loss = compute_loss(y_true, y_pred)
            losses.append(loss)

        # 3. Compute any additional metrics of each output.  (This could be
        #    done in the same loop as computing losses, but is done separately
        #    so timings can be compared directly.)
        crit_lists = []
        for i, (compute_crits, y_true, y_pred) in \
                enumerate(zip(compute_crit_lists, yy_true, yy_pred)):
            loss = Z.variable_to_numpy(losses[i])[0]
            crits = [loss]
            for compute_metric in compute_crits[1:]:
                metric = Z.variable_to_numpy(compute_metric(y_true, y_pred))[0]
                crits.append(metric)
            crit_lists.append(crits)

        # 4. Execute "after" callbacks.
        for callback in callbacks:
            callback.on_test_on_batch_end()

        return crit_lists

    def _fit_epoch(self, compute_crit_lists, dataset, optim, batch_size,
                   callbacks):
        for callback in callbacks:
            callback.on_epoch_begin(dataset.num_batches(batch_size))

        train_crit_lists = []
        test_crit_lists = []
        num_crits = 0
        for compute_crits in compute_crit_lists:
            train_crit_lists.append([[] for x in compute_crits])
            test_crit_lists.append([[] for x in compute_crits])
            num_crits += len(compute_crits)

        callback_names = [x.__class__.__name__ for x in callbacks]
        crit_name_lists = []
        for compute_crits in compute_crit_lists:
            crit_names = [x.__class__.__name__ for x in compute_crits]
            crit_name_lists.append(crit_names)

        train_timer = TrainBatchTimer(
            dataset.train.num_batches(batch_size), callback_names,
            crit_name_lists)

        for (xx, yy), is_training in dataset.each_batch(batch_size):
            xx = [Z.numpy_to_constant(x) for x in xx]
            yy = [Z.numpy_to_constant(y) for y in yy]
            if is_training:
                ret = self.train_on_batch(
                    xx, yy, compute_crit_lists, optim, callbacks, train_timer)
                split_crit_lists = train_crit_lists
            else:
                ret = self.test_on_batch(
                    xx, yy, compute_crit_lists, callbacks)
                split_crit_lists = test_crit_lists
            for i, values in enumerate(ret):
                for j, value in enumerate(values):
                    split_crit_lists[i][j].append(value)

        train_timer.summary()

        for split_crit_lists in [train_crit_lists, test_crit_lists]:
            for i, column in enumerate(split_crit_lists):
                for j, values in enumerate(column):
                    split_crit_lists[i][j] = float(np.mean(values))

        for callback in callbacks:
            callback.on_epoch_end()

        return train_crit_lists, test_crit_lists

    @require_kwargs_after(3)
    def fit(self, crit, data, test_frac=None, optim='sgd', batch=64,
            epoch_offset=0, epochs=10, callback=None):
        data = self._unpack_dataset(data, test_frac)
        y_sample_shapes = data.shapes()[0]
        crit_lists = self._unpack_crit_lists(crit, y_sample_shapes)
        optim = unpack_optim(optim)
        callbacks = unpack_callbacks(callback)
        assert isinstance(batch, int)
        assert 0 < batch
        assert isinstance(epoch_offset, int)
        assert 0 <= epoch_offset
        assert isinstance(epochs, int)
        assert 0 <= epochs
        optim.set_params(self.layer.params())
        for callback in callbacks:
            callback.on_fit_begin(epoch_offset, epochs)
        for epoch in range(epoch_offset, epoch_offset + epochs):
            train, test = \
                self._fit_epoch(crit_lists, data, optim, batch, callbacks)
            d = {
                'epoch': epoch,
                'crit': {
                    'train': train,
                    'test': test,
                },
            }
            print(json.dumps(d, indent=4, sort_keys=True))
        for callback in callbacks:
            callback.on_fit_end()

    @require_kwargs_after(2)
    def fit_reg(self, data, test_frac=None, optim='sgd', batch=64,
                epoch_offset=0, epochs=10, callback=None):
        crit = [['mean_squared_error']]
        return self.fit(crit, data, test_frac=test_frac, optim=optim,
                        batch=batch, epoch_offset=epoch_offset, epochs=epochs,
                        callback=callback)

    @require_kwargs_after(2)
    def fit_clf(self, data, test_frac=None, optim='sgd', batch=64,
                epoch_offset=0, epochs=10, callback=None):
        crit = [['cross_entropy', 'accuracy']]
        return self.fit(crit, data, test_frac=test_frac, optim=optim,
                        batch=batch, epoch_offset=epoch_offset, epochs=epochs,
                        callback=callback)
