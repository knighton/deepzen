import json
import numpy as np

from .. import backend as Z
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
    def _parse_crit_str(cls, s):
        ss = s.split(' ')
        return [s.split(',') for s in ss]

    @classmethod
    def _unpack_loss_and_metrics(cls, x, y_shapes):
        if not isinstance(x, (list, tuple)):
            x = [x]
        crits = []
        crits.append(unpack_loss(x[0], y_shapes))
        for item in x[1:]:
            crits.append(unpack_metric(item, y_shapes))
        return crits

    @classmethod
    def _unpack_crit(cls, x, y_shapes):
        if isinstance(x, str):
            if ' ' in x or ',' in x:
                x = cls._parse_crit_str(x)
            else:
                x = [x]
        crit = []
        for each in x:
            loss_and_metrics = \
                cls._unpack_loss_and_metrics(each, y_shapes)
            crit.append(loss_and_metrics)
        return crit

    def __init__(self, spec):
        self.spec = spec
        self.layer, self.out_form = spec.build()

    def forward(self, xx, is_training):
        x, = xx
        y_pred = self.layer.forward(x)
        return [y_pred]

    def train_on_batch(self, xx, yy_true, crit_lists, optim):
        losses = []
        with Z.autograd_record():
            yy_pred = self.forward(xx, True)
            for crits, y_true, y_pred in zip(crit_lists, yy_true, yy_pred):
                compute_loss = crits[0]
                loss = compute_loss(y_true, y_pred)
                losses.append(loss)
        grads = [Z.ones((1,), 'float32') for x in losses]
        Z.backward(losses, grads)
        optim.step()
        result_lists = []
        for i, (crits, y_true, y_pred) in \
                enumerate(zip(crit_lists, yy_true, yy_pred)):
            loss = Z.variable_to_numpy(losses[i])[0]
            results = [loss]
            for compute_metric in crits[1:]:
                metric = Z.variable_to_numpy(compute_metric(y_true, y_pred))[0]
                results.append(metric)
            result_lists.append(results)
        return result_lists

    def test_on_batch(self, xx, yy_true, crit_lists):
        yy_pred = self.forward(xx, False)
        result_lists = []
        for i, (crits, y_true, y_pred) in \
                enumerate(zip(crit_lists, yy_true, yy_pred)):
            results = []
            for compute_crit in crits:
                result = Z.variable_to_numpy(compute_crit(y_true, y_pred))[0]
                results.append(result)
            result_lists.append(results)
        return result_lists

    def _fit_epoch(self, crit_lists, dataset, optim, batch_size):
        train_results = []
        test_results = []
        for crits in crit_lists:
            train_results.append([[] for x in crits])
            test_results.append([[] for x in crits])

        for (xx, yy), is_training in dataset.each_batch(batch_size):
            xx = [Z.numpy_to_constant(x) for x in xx]
            yy = [Z.numpy_to_constant(y) for y in yy]
            if is_training:
                ret = self.train_on_batch(xx, yy, crit_lists, optim)
                split_results = train_results
            else:
                ret = self.test_on_batch(xx, yy, crit_lists)
                split_results = test_results
            for i, values in enumerate(ret):
                for j, value in enumerate(values):
                    split_results[i][j].append(value)

        for split_results in [train_results, test_results]:
            for i, column in enumerate(split_results):
                for j, values in enumerate(column):
                    split_results[i][j] = float(np.mean(values))

        return train_results, test_results

    @require_kwargs_after(3)
    def fit(self, crit, data, test_frac=None, optim='sgd', batch=64,
            epoch_offset=0, epochs=10):
        data = self._unpack_dataset(data, test_frac)
        y_shapes = data.shapes(batch)[0]
        crit = self._unpack_crit(crit, y_shapes)
        optim = unpack_optim(optim)
        assert isinstance(batch, int)
        assert 0 < batch
        assert isinstance(epoch_offset, int)
        assert 0 <= epoch_offset
        assert isinstance(epochs, int)
        assert 0 <= epochs
        optim.set_params(self.layer.params())
        for epoch in range(epoch_offset, epoch_offset + epochs):
            train, test = self._fit_epoch(crit, data, optim, batch)
            d = {
                'epoch': epoch,
                'train': train,
                'test': test,
            }
            print(json.dumps(d, indent=4, sort_keys=True))

    @require_kwargs_after(2)
    def fit_reg(self, data, test_frac=None, optim='sgd', batch=64,
                epoch_offset=0, epochs=10):
        crit = [['mean_squared_error']]
        return self.fit(crit, data, test_frac=test_frac, optim=optim,
                        batch=batch, epoch_offset=epoch_offset, epochs=epochs)

    @require_kwargs_after(2)
    def fit_clf(self, data, test_frac=None, optim='sgd', batch=64,
                epoch_offset=0, epochs=10):
        crit = [['categorical_cross_entropy', 'categorical_accuracy']]  # TODO
        return self.fit(crit, data, test_frac=test_frac, optim=optim,
                        batch=batch, epoch_offset=epoch_offset, epochs=epochs)
