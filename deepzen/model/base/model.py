import numpy as np

from ... import api as Z
from ...util.py import require_kwargs_after
from .trainer import Trainer


class Model(object):
    def __init__(self):
        self._is_built = False

    def is_built(self):
        """
        Get whether the model is built.
        """
        return self._is_built

    def build(self):
        """
        Build the model.
        """
        raise NotImplementedError

    def ensure_built(self):
        """
        Ensure that the model is built.
        """
        if self._is_built:
            return
        self.build()
        self._is_built = True

    def params(self):
        """
        Get all the model's parameters.
        """
        raise NotImplementedError

    def forward(self, xx, is_training):
        """
        Forward pass.
        """
        raise NotImplementedError

    def predict(self, xx, batch_size=64):
        if not isinstance(xx, (list, tuple)):
            xx = [xx]
        assert xx
        x = xx[0]
        num_batches = (len(x) + batch_size - 1) // batch_size
        batch_yyy = None
        for batch in range(num_batches):
            a = batch * batch_size
            z = (batch + 1) * batch_size
            batch_xx = [Z.constant(x[a:z]) for x in xx]
            batch_yy = self.forward(batch_xx, False)
            batch_yy = [Z.numpy(batch_y) for batch_y in batch_yy]
            if batch_yyy is None:
                batch_yyy = [[] for batch_y in batch_yy]
            for i, batch_y in enumerate(batch_yy):
                batch_yyy[i].append(batch_y)
        return [np.concatenate(batch_yy, 0) for batch_yy in batch_yyy]

    # --------------------------------------------------------------------------
    # Calling fit on a single batch.

    def train_on_batch(self, xx, yy_true, optimizer, losses, aux_meter_lists):
        costs = []
        with Z.autograd_record():
            yy_pred = self.forward(xx, True)

            for loss, y_true, y_pred in zip(losses, yy_true, yy_pred):
                cost = Z.mean(loss(y_true, y_pred))
                costs.append(cost)

        grads = [Z.ones((1,), Z.dtype(x), Z.device(x)) for x in costs]
        Z.backward(costs, grads)

        costs = [Z.scalar(x) for x in costs]

        optimizer.step()

        aux_metric_lists = []
        for meters, y_true, y_pred in zip(aux_meter_lists, yy_true, yy_pred):
            metrics = []
            for meter in meters:
                metric = Z.scalar(Z.mean(meter(y_true, y_pred)))
                metrics.append(metric)
            aux_metric_lists.append(metrics)

        return costs, aux_metric_lists

    def test_on_batch(self, xx, yy_true, losses, aux_meter_lists):
        yy_pred = self.forward(xx, False)

        costs = []
        for loss, y_true, y_pred in zip(losses, yy_true, yy_pred):
            cost = Z.mean(loss(y_true, y_pred))
            costs.append(cost)

        costs = [Z.scalar(x) for x in costs]

        aux_metric_lists = []
        for meters, y_true, y_pred in zip(aux_meter_lists, yy_true, yy_pred):
            metrics = []
            for meter in meters:
                metric = Z.scalar(Z.mean(meter(y_true, y_pred)))
                metrics.append(metric)
            aux_metric_lists.append(metrics)

        return costs, aux_metric_lists

    def fit_on_batch(self, is_training, xx, yy_true, optimizer, losses,
                     aux_meter_lists):
        xx = [Z.constant(x) for x in xx]
        yy_true = [Z.constant(y) for y in yy_true]
        if is_training:
            ret = self.train_on_batch(xx, yy_true, optimizer, losses,
                                      aux_meter_lists)
        else:
            ret = self.test_on_batch(xx, yy_true, losses, aux_meter_lists)
        return ret

    # --------------------------------------------------------------------------
    # Train on batch.

    def fit_train_on_batch(self, trainer, xx, yy_true):
        """
        Train on one batch.
        """
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
            spy.on_train_on_batch_end(metric_lists)
            t.mark()
        t.mark()

        # Stop timing the whole method.
        t.stop()

        return metric_lists

    # --------------------------------------------------------------------------
    # Test on batch.

    def fit_test_on_batch(self, trainer, xx, yy_true):
        """
        Test on one batch.
        """
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

        # 4. Compute any additional metrics of each output.  (This could be done
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

        # 5. Execute "on end" callbacks.
        t.mark()
        for spy in trainer.spies:
            t.mark()
            spy.on_test_on_batch_end(metric_lists)
            t.mark()
        t.mark()

        # Stop timing the whole method.
        t.stop()

        return metric_lists

    # --------------------------------------------------------------------------
    # Fit on batch, calling batch train/test (notes epochs when appropriate).

    def _fit_on_epoch_before(self, trainer):
        """
        Internally, note that we've begun a training epoch.
        """
        for spy in trainer.spies:
            spy.on_epoch_begin()

    def _fit_on_epoch_after(self, trainer, epoch_results):
        """
        Internally, note that we've ended a training epoch.
        """
        for spy in trainer.spies:
            spy.on_epoch_end(epoch_results)

    def resume_fit_batch(self, trainer, is_training, xx, yy):
        """
        Fit (train or test) on one batch.
        """
        if not trainer.cursor.batch:
            self._fit_on_epoch_before(trainer)

        xx = [Z.constant(x) for x in xx]
        yy = [Z.constant(y) for y in yy]
        if is_training:
            results = self.fit_train_on_batch(trainer, xx, yy)
        else:
            results = self.fit_test_on_batch(trainer, xx, yy)

        trainer.epoch_results.add(is_training, results)
        is_epoch_done, is_fit_done = \
            trainer.cursor.note_completed_batch(is_training)
        if is_epoch_done:
            epoch_results = trainer.epoch_results.harvest()
            self._fit_on_epoch_after(trainer, epoch_results)
        return is_fit_done

    # --------------------------------------------------------------------------
    # Fit given a training state.

    def resume_fit(self, trainer):
        """
        Fit the model, according to the training state.
        """
        for spy in trainer.spies:
            spy.set_params(self, trainer)
            spy.on_fit_begin()

        each_batch_forever = trainer.dataset.each_batch_forever
        batch_size = trainer.cursor.batch_size
        for (xx, yy), is_training in each_batch_forever(batch_size):
            if self.resume_fit_batch(trainer, is_training, xx, yy):
                break

        for spy in trainer.spies:
            spy.on_fit_end()

    # --------------------------------------------------------------------------
    # Fit with smart arguments.

    @require_kwargs_after(3)
    def fit(self, data, loss, test_frac=None, optim='adam', batch=64, start=0,
            stop=20, spy=None, timer_cache=10000):
        """
        Fit the model, according to the smart arguments (creates a trainer).
        """
        trainer = Trainer.init_from_args(
            data, loss, test_frac, optim, batch, start, stop, spy, timer_cache)
        self.ensure_built()
        trainer.optimizer.set_params(self.params())  # TODO: fix this.
        self.resume_fit(trainer)
        return trainer

    @require_kwargs_after(2)
    def fit_reg(self, data, test_frac=None, optim='adam', batch=64, start=0,
                stop=20, spy=None, timer_cache=10000):
        """
        Fit the model as a regressor with one output.
        """
        loss = [['mean_squared_error']]
        return self.fit(data, loss, test_frac=test_frac, optim=optim,
                        batch=batch, start=start, stop=stop, spy=spy,
                        timer_cache=timer_cache)

    @require_kwargs_after(2)
    def fit_clf(self, data, test_frac=None, optim='adam', batch=64, start=0,
                stop=20, spy=None, timer_cache=10000):
        """
        Fit the model as a classifier with one output.
        """
        loss = [['cross_entropy', 'accuracy']]
        return self.fit(data, loss, test_frac=test_frac, optim=optim,
                        batch=batch, start=start, stop=stop, spy=spy,
                        timer_cache=timer_cache)
