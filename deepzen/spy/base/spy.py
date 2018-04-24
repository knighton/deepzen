class Spy(object):
    """
    Monitors model training.
    """

    def set_params(self, model, session):
        self.model = model
        self.session = session

    def on_fit_begin(self):
        pass

    def on_epoch_begin(self):
        pass

    def on_train_on_batch_begin(self):
        pass

    def on_train_on_batch_end(self, metric_lists):
        pass

    def on_test_on_batch_begin(self):
        pass

    def on_test_on_batch_end(self, metric_lists):
        pass

    def on_epoch_end(self, epoch_results):
        pass

    def on_fit_end(self):
        pass
