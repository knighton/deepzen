from ...base.metric import BaseMetricAPI
from .accuracy import PyTorchAccuracyAPI
from .loss import PyTorchLossAPI


class PyTorchMetricAPI(BaseMetricAPI, PyTorchAccuracyAPI, PyTorchLossAPI):
    pass
