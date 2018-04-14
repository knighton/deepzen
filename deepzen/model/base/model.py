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
        num_batches = (len(xx) + batch_size - 1) // batch_size
        batch_yyy = []
        for batch in range(num_batches):
            batch_xx = xx[batch * batch_size : (batch + 1) * batch_size]
            batch_xx = [Z.constant(x) for x in batch_xx]
            batch_yy = self.forward(batch_xx, False)
            batch_yy = [Z.numpy(batch_y) for batch_y in batch_yy]
            batch_yyy.append(batch_yy)
        return np.concatenate(batch_yyy, 0)

    # --------------------------------------------------------------------------
    # Train on batch.

    def train_on_batch(self, trainer, xx, yy_true):
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

    def test_on_batch(self, trainer, xx, yy_true):
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

    def fit_on_batch_before(self, trainer):
        """
        The pre-work we need to run before the body of fit_on_batch.
        """
        if not trainer.cursor.batch:
            self._fit_on_epoch_before(trainer)

    def fit_on_batch_body(self, trainer, is_training, xx, yy):
        """
        The work of fit_on_batch, bookended by pre/post callbacks.
        """
        xx = [Z.constant(x) for x in xx]
        yy = [Z.constant(y) for y in yy]
        if is_training:
            results = self.train_on_batch(trainer, xx, yy)
        else:
            results = self.test_on_batch(trainer, xx, yy)
        return results

    def fit_on_batch_after(self, trainer, is_training, xx, yy, results):
        """
        The post-work we need to run after the body of fit_on_batch.
        """
        trainer.epoch_results.add(is_training, results)
        is_epoch_done, is_fit_done = \
            trainer.cursor.note_completed_batch(is_training)
        if is_epoch_done:
            epoch_results = trainer.epoch_results.harvest()
            self._fit_on_epoch_after(trainer, epoch_results)
        return is_fit_done

    def fit_on_batch(self, trainer, is_training, xx, yy):
        """
        Fit (train or test) on one batch.
        """
        self.fit_on_batch_before(trainer)
        results = self.fit_on_batch_body(trainer, is_training, xx, yy)
        return self.fit_on_batch_after(trainer, is_training, xx, yy, results)

    # --------------------------------------------------------------------------
    # Fit given a training state.

    def fit_before(self, trainer):
        """
        The pre-work we need to run before the body of resume_fit.
        """
        for spy in trainer.spies:
            spy.set_params(self, trainer)
            spy.on_fit_begin()

    def fit_body(self, trainer):
        """
        The work of resume_fit, bookended by pre/post callbacks.
        """
        each_batch_forever = trainer.dataset.each_batch_forever
        batch_size = trainer.cursor.batch_size
        for (xx, yy), is_training in each_batch_forever(batch_size):
            if self.fit_on_batch(trainer, is_training, xx, yy):
                break

    def fit_after(self, trainer):
        """
        The post-work we need to run after the body of resume_fit.
        """
        for spy in trainer.spies:
            spy.on_fit_end()

    def resume_fit(self, trainer):
        """
        Fit the model, according to the training state.
        """
        self.fit_before(trainer)
        self.fit_body(trainer)
        self.fit_after(trainer)

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
