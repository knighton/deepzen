class BaseAccuracyAPI(object):
    def binary_accuracy(self, true, pred):
        shape = self.shape(pred)
        assert len(shape) == 2
        assert shape[1] == 1
        pred = self.round(pred)
        hits = self.equal(true, pred)
        return self.mean(hits, -1, False)

    def categorical_accuracy(self, true, pred):
        true_indices = self.argmax(true, -1)
        pred_indices = self.argmax(pred, -1)
        hits = self.equal(true_indices, pred_indices)
        hits = self.cast(hits, self.dtype(true))
        return self.mean(hits, -1, False)
