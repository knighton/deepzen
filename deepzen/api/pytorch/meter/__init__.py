from ...base.meter import BaseMeterAPI
from .accuracy import PyTorchAccuracyAPI
from .loss import PyTorchLossAPI


class PyTorchMeterAPI(BaseMeterAPI, PyTorchAccuracyAPI, PyTorchLossAPI):
    pass
