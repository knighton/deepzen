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

    def on_train_on_batch_begin(self, session):
        t = session.batch_timer.train
        t.mark()
        for spy in session.spies:
            t.mark()
            spy.on_train_on_batch_begin()
            t.mark()
        t.mark()

    def on_train_on_batch_end(self, session):
        t = session.batch_timer.train
        t.mark()
        for spy in session.spies:
            t.mark()
            spy.on_train_on_batch_end()
            t.mark()
        t.mark()

    def train_on_batch(self, session, xx, yy_true):
        # Start timing the whole method.
        t = session.batch_timer.train
        t.start()

        # 1. Execute "on begin" callbacks.
        self.on_train_on_batch_begin(session)

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
        self.on_train_on_batch_end(session)

        # Stop timing the whole method.
        t.stop()

        return metric_lists

    def on_test_on_batch_begin(self, session):
        t = session.batch_timer.test
        t.mark()
        for spy in session.spies:
            t.mark()
            spy.on_test_on_batch_begin()
            t.mark()
        t.mark()

    def on_test_on_batch_end(self, session):
        t = session.batch_timer.test
        t.mark()
        for spy in session.spies:
            t.mark()
            spy.on_test_on_batch_end()
            t.mark()
        t.mark()

    def test_on_batch(self, session, xx, yy_true):
        # Start timing the whole method.
        t = session.batch_timer.test
        t.start()

        # 1. Execute "on begin" callbacks.
        self.on_test_on_batch_begin(session)

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
        self.on_test_on_batch_end(session)

        # Stop timing the whole method.
        t.stop()

        return metric_lists

    def on_epoch_begin(self, session):
        for spy in session.spies:
            spy.on_epoch_begin(session.cursor.epoch,
                               session.cursor.batches_per_epoch)

    def on_epoch_end(self, session, results):
        raws, means = results
        train_metric_lists, test_metric_lists = means
        for spy in session.spies:
            spy.on_epoch_end(train_metric_lists, test_metric_lists)

    def _fit_batch(self, session, is_training, xx, yy):
        xx = [Z.constant(x) for x in xx]
        yy = [Z.constant(y) for y in yy]

        if not session.cursor.batch:
            self.on_epoch_begin(session)

        if is_training:
            batch_metric_lists = self.train_on_batch(session, xx, yy)
        else:
            batch_metric_lists = self.test_on_batch(session, xx, yy)

        is_epoch_done, is_fit_done = session.note_batch_results(
            is_training, batch_metric_lists)

        if is_epoch_done:
            epoch_results = session.collector.harvest()
            self.on_epoch_end(session, epoch_results)

        return is_fit_done

    def on_fit_begin(self, session):
        for spy in session.spies:
            spy.on_fit_begin(session.batch_timer.meter_name_lists,
                             session.cursor.epoch, session.cursor.end_epoch)

    def on_fit_end(self, session):
        for spy in session.spies:
            spy.on_fit_end()

    def fit_session(self, session):
        self.on_fit_begin(session)
        for (xx, yy), is_training in session.each_batch():
            if self._fit_batch(session, is_training, xx, yy):
                break
        self.on_fit_end(session)
        return session

    @require_kwargs_after(3)
    def fit(self, data, loss, test_frac=None, optim='adam', batch=64, start=0,
            stop=20, spy=None, timer_cache=10000):
        session = TrainingSession.init_from_args(
            data, loss, test_frac, optim, batch, start, stop, spy, timer_cache)
        self.ensure_built()
        session.optimizer.set_params(self.params())  # TODO: fix this.
        return self.fit_session(session)

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
