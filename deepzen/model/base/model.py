from ... import api as Z
from ...util.py import require_kwargs_after
from .training_session import TrainingSession


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

    # --------------------------------------------------------------------------
    # Train on batch.

    def train_on_batch(self, session, xx, yy_true):
        # Start timing the whole method.
        t = session.batch_timer.train
        t.start()

        # 1. Execute "on begin" callbacks.
        t.mark()
        for spy in session.spies:
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
                    zip(session.meter_lists, yy_true, yy_pred):
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
                enumerate(zip(session.meter_lists, yy_true, yy_pred)):
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
        session.optimizer.step()
        t.mark()

        # 7. Execute "on end" callbacks.
        t.mark()
        for spy in session.spies:
            t.mark()
            spy.on_train_on_batch_end()
            t.mark()
        t.mark()

        # Stop timing the whole method.
        t.stop()

        return metric_lists

    # --------------------------------------------------------------------------
    # Test on batch.

    def test_on_batch(self, session, xx, yy_true):
        # Start timing the whole method.
        t = session.batch_timer.test
        t.start()

        # 1. Execute "on begin" callbacks.
        t.mark()
        for spy in session.spies:
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
                enumerate(zip(session.meter_lists, yy_true, yy_pred)):
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
                enumerate(zip(session.meter_lists, yy_true, yy_pred)):
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
        for spy in session.spies:
            t.mark()
            spy.on_test_on_batch_end()
            t.mark()
        t.mark()

        # Stop timing the whole method.
        t.stop()

        return metric_lists

    # --------------------------------------------------------------------------
    # Fit on batch, calling batch train/test (bookends epochs when appropriate).

    def _fit_epoch_before(self, session):
        for spy in session.spies:
            spy.on_epoch_begin(session.cursor.epoch,
                               session.cursor.batches_per_epoch)

    def _fit_epoch_after(self, session, results):
        raws, means = results
        train_metric_lists, test_metric_lists = means
        for spy in session.spies:
            spy.on_epoch_end(train_metric_lists, test_metric_lists)

    def fit_batch_before(self, session):
        if not session.cursor.batch:
            self._fit_epoch_before(session)

    def fit_batch_main(self, session, is_training, xx, yy):
        xx = [Z.constant(x) for x in xx]
        yy = [Z.constant(y) for y in yy]
        if is_training:
            batch_metric_lists = self.train_on_batch(session, xx, yy)
        else:
            batch_metric_lists = self.test_on_batch(session, xx, yy)
        return batch_metric_lists

    def fit_batch_after(self, session, is_training, xx, yy, batch_metric_lists):
        session.collector.add(is_training, batch_metric_lists)
        is_epoch_done, is_fit_done = \
            session.cursor.note_completed_batch(is_training)
        if is_epoch_done:
            epoch_results = session.collector.harvest()
            self._fit_epoch_after(session, epoch_results)
        return is_fit_done

    def fit_batch(self, session, is_training, xx, yy):
        self.fit_batch_before(session)
        batch_metric_lists = self.fit_batch_main(session, is_training, xx, yy)
        return self.fit_batch_after(session, is_training, xx, yy,
                                    batch_metric_lists)

    # --------------------------------------------------------------------------
    # Fit given a training session.

    def fit_before(self, session):
        for spy in session.spies:
            spy.on_fit_begin(session.batch_timer.meter_name_lists,
                             session.cursor.epoch, session.cursor.end_epoch)

    def _each_batch_forever(self, session):
        while True:
            for batch in session.dataset.each_batch(session.cursor.batch_size):
                yield batch

    def fit_main(self, session):
        for (xx, yy), is_training in self._each_batch_forever(session):
            if self.fit_batch(session, is_training, xx, yy):
                break

    def fit_after(self, session):
        for spy in session.spies:
            spy.on_fit_end()

    def fit_session(self, session):
        self.fit_before(session)
        self.fit_main(session)
        self.fit_after(session)

    # --------------------------------------------------------------------------
    # Fit with smart arguments.

    @require_kwargs_after(3)
    def fit(self, data, loss, test_frac=None, optim='adam', batch=64, start=0,
            stop=20, spy=None, timer_cache=10000):
        session = TrainingSession.init_from_args(
            data, loss, test_frac, optim, batch, start, stop, spy, timer_cache)
        self.ensure_built()
        session.optimizer.set_params(self.params())  # TODO: fix this.
        self.fit_session(session)
        return session

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
