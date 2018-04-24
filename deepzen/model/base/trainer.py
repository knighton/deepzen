from ...data import unpack_dataset
from ...spy import unpack_spies
from ...optim import unpack_optimizer
from ...meter import unpack_meter_lists
from .batch_timer import BatchTimer
from .metric_collector import MetricCollector
from .training_cursor import TrainingCursor


class Trainer(object):
    def __init__(self, dataset, cursor, epoch_results, meter_lists, optimizer,
                 spies, batch_timer):
        self.dataset = dataset
        self.cursor = cursor
        self.epoch_results = epoch_results
        self.meter_lists = meter_lists
        self.optimizer = optimizer
        self.spies = spies
        self.batch_timer = batch_timer

    @classmethod
    def init_from_args(cls, data, loss, test_frac=None, optimizer='adam',
                       batch_size=64, begin_epoch=0, end_epoch=20, spy=None,
                       timer_cache_size=10000):
        dataset = unpack_dataset(data, test_frac)
        meter_lists = unpack_meter_lists(loss, dataset.y_sample_shapes)
        optimizer = unpack_optimizer(optimizer)
        spies = unpack_spies(spy)
        assert isinstance(batch_size, int)
        assert 0 < batch_size
        assert isinstance(begin_epoch, int)
        assert isinstance(end_epoch, int)
        assert 0 <= begin_epoch <= end_epoch
        assert isinstance(timer_cache_size, int)
        assert 0 < timer_cache_size

        cursor = TrainingCursor.start_from_dataset(
            dataset, begin_epoch, end_epoch, batch_size)
        epoch_results = MetricCollector.start_from_meter_lists(meter_lists)
        batch_timer = BatchTimer.init_getting_names(
            timer_cache_size, spies, meter_lists)

        return cls(dataset, cursor, epoch_results, meter_lists, optimizer,
                   spies, batch_timer)
