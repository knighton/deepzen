import numpy as np

from .. import api as Z
from ..hook import get_hooks
from ..io.dataset import Dataset
from ..io.ram_split import RamSplit
from ..io.split import Split
from ..scorer.loss import get_loss_scorer
from ..scorer import get_scorer
from ..optim import get_optimizer
from ..util.py import require_kwargs_after
from .batch_timer import IntraBatchTimer


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
    def _parse_scorer_lists_str(cls, s):
        ss = s.split(' ')
        return [s.split(',') for s in ss]

    @classmethod
    def _get_loss_and_aux_scorers(cls, x, y_sample_shape):
        if isinstance(x, (list, tuple)):
            xx = x
        else:
            xx = [x]
        scorers = []
        scorers.append(get_loss_scorer(xx[0], y_sample_shape))
        for x in xx[1:]:
            scorers.append(get_scorer(x, y_sample_shape))
        return scorers

    @classmethod
    def _get_scorer_lists(cls, x, y_sample_shapes):
        if isinstance(x, str):
            if ' ' in x or ',' in x:
                xxx = cls._parse_scorer_lists_str(x)
            else:
                xxx = [x]
        else:
            xxx = x
        scorer_lists = []
        assert len(xxx) == len(y_sample_shapes)
        for xx, y_sample_shape in zip(xxx, y_sample_shapes):
            scorers = cls._get_loss_and_aux_scorers(xx, y_sample_shape)
            scorer_lists.append(scorers)
        return scorer_lists

    def __init__(self, spec):
        self.spec = spec
        self.layer = spec.build()

    def forward(self, xx, is_training):
        x, = xx
        y_pred = self.layer.forward(x, is_training)
        return [y_pred]

    def train_on_batch(self, xx, yy_true, scorer_lists, optim, hooks, t):
        # Start timing the whole method.
        t.start()

        # 1. Execute "on begin" hooks.
        t.mark()
        for hook in hooks:
            t.mark()
            hook.on_train_on_batch_begin()
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
            for scorers, y_true, y_pred in \
                    zip(scorer_lists, yy_true, yy_pred):
                get_loss = scorers[0]
                t.mark()
                loss = Z.mean(get_loss(y_true, y_pred))
                t.mark()
                losses.append(loss)
            t.mark()

        # 4. Compute any additional scores of each output.
        score_lists = []
        t.mark()
        for i, (scorers, y_true, y_pred) in \
                enumerate(zip(scorer_lists, yy_true, yy_pred)):
            loss = Z.scalar(losses[i])
            scores = [loss]
            for aux_scorer in scorers[1:]:
                t.mark()
                score = Z.mean(aux_scorer(y_true, y_pred))
                t.mark()
                score = Z.scalar(score)
                scores.append(score)
            score_lists.append(scores)
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

        # 7. Execute "on end" hooks.
        t.mark()
        for hook in hooks:
            t.mark()
            hook.on_train_on_batch_end()
            t.mark()
        t.mark()

        # Stop timing the whole method.
        t.stop()

        return score_lists

    def test_on_batch(self, xx, yy_true, scorer_lists, hooks, t):
        # Start timing the whole method.
        t.start()

        # 1. Execute "before" hooks.
        t.mark()
        for hook in hooks:
            t.mark()
            hook.on_test_on_batch_begin()
            t.mark()
        t.mark()

        # 2. Forward propagate.
        t.mark()
        yy_pred = self.forward(xx, False)
        t.mark()

        # 3. Compute the loss of each output.
        losses = []
        t.mark()
        for i, (scorers, y_true, y_pred) in \
                enumerate(zip(scorer_lists, yy_true, yy_pred)):
            get_loss = scorers[0]
            t.mark()
            loss = Z.mean(get_loss(y_true, y_pred))
            t.mark()
            losses.append(loss)
        t.mark()

        # 3. Compute any additional scores of each output.  (This could be done
        #    in the same loop as computing losses, but is done separately so
        #    timings can be compared directly.)
        score_lists = []
        t.mark()
        for i, (scorers, y_true, y_pred) in \
                enumerate(zip(scorer_lists, yy_true, yy_pred)):
            loss = Z.scalar(losses[i])
            scores = [loss]
            for aux_scorer in scorers[1:]:
                t.mark()
                score = Z.mean(aux_scorer(y_true, y_pred))
                t.mark()
                score = Z.scalar(score)
                scores.append(score)
            score_lists.append(scores)
        t.mark()

        # 4. Execute "after" hooks.
        t.mark()
        for hook in hooks:
            t.mark()
            hook.on_test_on_batch_end()
            t.mark()
        t.mark()

        # Stop timing the whole method.
        t.stop()

        return score_lists

    def _fit_epoch(self, scorer_lists, dataset, optim, batch_size, hooks, timer,
                   epoch):
        for hook in hooks:
            hook.on_epoch_begin(epoch, dataset.num_batches(batch_size))

        train_score_lists = []
        test_score_lists = []
        for scorers in scorer_lists:
            train_score_lists.append([[] for x in scorers])
            test_score_lists.append([[] for x in scorers])

        for (xx, yy), is_training in dataset.each_batch(batch_size):
            xx = [Z.constant(x) for x in xx]
            yy = [Z.constant(y) for y in yy]
            if is_training:
                batch_score_lists = self.train_on_batch(
                    xx, yy, scorer_lists, optim, hooks, timer.train)
                split_score_lists = train_score_lists
            else:
                batch_score_lists = self.test_on_batch(
                    xx, yy, scorer_lists, hooks, timer.test)
                split_score_lists = test_score_lists
            for i, batch_scores in enumerate(batch_score_lists):
                for j, batch_score in enumerate(batch_scores):
                    split_score_lists[i][j].append(batch_score)

        for split_score_lists in [train_score_lists, test_score_lists]:
            for i, column in enumerate(split_score_lists):
                for j, values in enumerate(column):
                    split_score_lists[i][j] = float(np.mean(values))

        for hook in hooks:
            hook.on_epoch_end(train_score_lists, test_score_lists)

        return train_score_lists, test_score_lists

    @require_kwargs_after(3)
    def fit(self, data, loss, test_frac=None, optim='sgd', batch=64,
            epoch_offset=0, epochs=20, hook=None, timer_cache=10000):
        dataset = self._unpack_dataset(data, test_frac)
        y_sample_shapes = dataset.shapes()[0]
        scorer_lists = self._get_scorer_lists(loss, y_sample_shapes)
        optim = get_optimizer(optim)
        hooks = get_hooks(hook)
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

        hook_names = [x.__class__.__name__ for x in hooks]
        scorer_name_lists = []
        for scorers in scorer_lists:
            scorer_names = [x.__class__.__name__ for x in scorers]
            scorer_name_lists.append(scorer_names)
        timer = IntraBatchTimer(timer_cache_size, hook_names, scorer_name_lists)

        optim.set_params(self.layer.params())

        for hook in hooks:
            hook.on_fit_begin(scorer_name_lists, epoch_offset, epochs)

        for epoch in range(epoch_offset, epoch_offset + epochs):
            train_score_lists, test_score_lists = \
                self._fit_epoch(scorer_lists, dataset, optim, batch_size, hooks,
                                timer, epoch)

        for hook in hooks:
            hook.on_fit_end()

    @require_kwargs_after(2)
    def fit_reg(self, data, test_frac=None, optim='sgd', batch=64,
                epoch_offset=0, epochs=20, hook=None, timer_cache=10000):
        loss = [['mean_squared_error']]
        return self.fit(data, loss, test_frac=test_frac, optim=optim,
                        batch=batch, epoch_offset=epoch_offset, epochs=epochs,
                        hook=hook, timer_cache=timer_cache)

    @require_kwargs_after(2)
    def fit_clf(self, data, test_frac=None, optim='sgd', batch=64,
                epoch_offset=0, epochs=20, hook=None, timer_cache=10000):
        loss = [['cross_entropy', 'accuracy']]
        return self.fit(data, loss, test_frac=test_frac, optim=optim,
                        batch=batch, epoch_offset=epoch_offset, epochs=epochs,
                        hook=hook, timer_cache=timer_cache)
