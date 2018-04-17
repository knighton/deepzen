import numpy as np


class MetricCollector(object):
    """
    Accumulates batch metric list results; flushed at the end of every epoch.
    """

    def __init__(self, train, test):
        self.train = train
        self.test = test

    @classmethod
    def start(cls, metrics_per_output):
        train = []
        test = []
        for num_metrics in metrics_per_output:
            train.append([[] for metric in range(num_metrics)])
            test.append([[] for metric in range(num_metrics)])
        return cls(train, test)

    @classmethod
    def start_from_meter_lists(cls, meter_lists):
        metrics_per_output = list(map(len, meter_lists))
        return cls.start(metrics_per_output)

    def add(self, is_training, batch_metric_lists):
        if is_training:
            split = self.train
        else:
            split = self.test
        for i, batch_metrics in enumerate(batch_metric_lists):
            for j, batch_metric in enumerate(batch_metrics):
                split[i][j].append(batch_metric)

    @classmethod
    def harvest_samples(cls, metric_lists_samples):
        metric_lists = []
        for i, metrics_samples in enumerate(metric_lists_samples):
            metrics = []
            for j, metric_samples in enumerate(metrics_samples):
                metric = float(np.mean(metric_samples))
                metric_samples.clear()
                metrics.append(metric)
            metric_lists.append(metrics)
        return metric_lists

    def harvest(self):
        raw_samples = self.train, self.test
        train = self.harvest_samples(self.train)
        test = self.harvest_samples(self.test)
        means = train, test
        return raw_samples, means
