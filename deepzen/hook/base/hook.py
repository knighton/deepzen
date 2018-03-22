class Hook(object):
    def on_fit_begin(self, scorer_name_lists, epoch_offset, num_epochs):
        pass

    def on_fit_end(self):
        pass

    def on_epoch_begin(self, epoch, num_batches):
        pass

    def on_epoch_end(self, train_score_lists, test_score_lists):
        pass

    def on_train_on_batch_begin(self):
        pass

    def on_train_on_batch_end(self):
        pass

    def on_test_on_batch_begin(self):
        pass

    def on_test_on_batch_end(self):
        pass
