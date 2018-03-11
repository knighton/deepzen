import json
import numpy as np

from .. import backend as Z
from ..iter.dataset import Dataset
from ..iter.ram_split import RamSplit


class Model(object):
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

    def fit(self, optim, loss_and_metrics, dataset, epochs, batch_size):
        (x_train, y_train), (x_test, y_test) = dataset
        train = RamSplit(x_train, y_train)
        test = RamSplit(x_test, y_test)
        dataset = Dataset(train, test)
        optim.set_params(self.layer.params())
        crit_lists = [loss_and_metrics]
        for epoch in range(epochs):
            train, test = self._fit_epoch(
                crit_lists, dataset, optim, batch_size)
            x = {
                'epoch': epoch,
                'train': train,
                'test': test,
            }
            print(json.dumps(x, indent=4, sort_keys=True))
