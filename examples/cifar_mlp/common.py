import numpy as np


class SplitResults(object):
    def __init__(self):
        self.losses = []
        self.accs = []
        self.times = []

    def add(self, costs, aux_metric_lists, t):
        loss, = costs
        tmp, = aux_metric_lists
        acc, = tmp
        self.losses.append(loss)
        self.accs.append(acc)
        self.times.append(t)

    def summary(self):
        loss = float(np.mean(self.losses))
        acc = float(np.mean(self.accs))
        time = float(np.mean(self.times))
        return {
            'loss': loss,
            'acc': acc,
            'time': time,
        }


class EpochResults(object):
    def __init__(self):
        self.train = SplitResults()
        self.eval = SplitResults()

    def add(self, is_training, costs, aux_metric_lists, t):
        if is_training:
            split = self.train
        else:
            split = self.eval
        split.add(costs, aux_metric_lists, t)

    def summary(self):
        return {
            'train': self.train.summary(),
            'eval': self.eval.summary(),
        }
