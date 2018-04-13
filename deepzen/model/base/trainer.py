class Trainer(object):
    """
    For training, the goal, how to get there, and monitoring.
    """

    def __init__(self, meter_lists, optimizer, spies, batch_timer):
        self.meter_lists = meter_lists
        self.optimizer = optimizer
        self.spies = spies
        self.batch_timer = batch_timer
