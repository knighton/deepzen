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

    def train_on_batch(self, xx, yy_true, compute_crit_lists, optim, callbacks):
        for callback in callbacks:
            callback.on_train_on_batch_begin()
        losses = []
        with Z.autograd_record():
            t0 = time()
            yy_pred = self.forward(xx, True)
            t_forward = time() - t0
            for compute_crits, y_true, y_pred in \
                    zip(compute_crit_lists, yy_true, yy_pred):
                compute_loss = compute_crits[0]
                loss = compute_loss(y_true, y_pred)
                losses.append(loss)
        grads = [Z.ones((1,), 'float32') for x in losses]
        t0 = time()
        Z.backward(losses, grads)
        t_backward = time() - t0
        t0 = time()
        optim.step()
        t_optim = time() - t0
        crit_lists = []
        for i, (compute_crits, y_true, y_pred) in \
                enumerate(zip(compute_crit_lists, yy_true, yy_pred)):
            loss = Z.variable_to_numpy(losses[i])[0]
            crits = [loss]
            for compute_metric in compute_crits[1:]:
                metric = Z.variable_to_numpy(compute_metric(y_true, y_pred))[0]
                crits.append(metric)
            crit_lists.append(crits)
        times = t_forward, t_backward, t_optim
        for callback in callbacks:
            callback.on_train_on_batch_end()
        return crit_lists, times

    def test_on_batch(self, xx, yy_true, compute_crit_lists, callbacks):
        for callback in callbacks:
            callback.on_test_on_batch_begin()
        t0 = time()
        yy_pred = self.forward(xx, False)
        t_forward = time() - t0
        crit_lists = []
        for i, (compute_crits, y_true, y_pred) in \
                enumerate(zip(compute_crit_lists, yy_true, yy_pred)):
            crits = []
            for compute_crit in compute_crits:
                crit = Z.variable_to_numpy(compute_crit(y_true, y_pred))[0]
                crits.append(crit)
            crit_lists.append(crits)
        times = t_forward,
        for callback in callbacks:
            callback.on_test_on_batch_end()
        return crit_lists, times

    def _fit_epoch(self, crit_lists, dataset, optim, batch_size, callbacks):
        for callback in callbacks:
            callback.on_epoch_begin(dataset.num_batches(batch_size))

        train_results = []
        test_results = []
        for crits in crit_lists:
            train_results.append([[] for x in crits])
            test_results.append([[] for x in crits])
        t_train_forward = []
        t_train_backward = []
        t_train_optim = []
        t_test_forward = []

        for (xx, yy), is_training in dataset.each_batch(batch_size):
            xx = [Z.numpy_to_constant(x) for x in xx]
            yy = [Z.numpy_to_constant(y) for y in yy]
            if is_training:
                t0 = time()
                ret, times = self.train_on_batch(
                    xx, yy, crit_lists, optim, callbacks)
                t = time() - t0
                t_forward, t_backward, t_optim = times
                split_results = train_results
                t_train_forward.append(t_forward)
                t_train_backward.append(t_backward)
                t_train_optim.append(t_optim)
            else:
                ret, times = self.test_on_batch(xx, yy, crit_lists, callbacks)
                split_results = test_results
                t_forward, = times
                t_test_forward.append(t_forward)
            for i, values in enumerate(ret):
                for j, value in enumerate(values):
                    split_results[i][j].append(value)

        for split_results in [train_results, test_results]:
            for i, column in enumerate(split_results):
                for j, values in enumerate(column):
                    split_results[i][j] = float(np.mean(values))

        t_train_forward = float(np.mean(t_train_forward))
        t_train_backward = float(np.mean(t_train_backward))
        t_train_optim = float(np.mean(t_train_optim))
        t_test_forward = float(np.mean(t_test_forward))
        times = {
            'train_forward': t_train_forward,
            'train_backward': t_train_backward,
            'train_optim': t_train_optim,
            'test_forward': t_test_forward,
        }

        for callback in callbacks:
            callback.on_epoch_end()

        return (train_results, test_results), times

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
            (train, test), times = \
                self._fit_epoch(crit_lists, data, optim, batch, callbacks)
            d = {
                'epoch': epoch,
                'train': train,
                'test': test,
                'time': times,
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
