from ...base.score import BaseScoreAPI
from .accuracy import PyTorchAccuracyAPI
from .loss import PyTorchLossAPI


class PyTorchScoreAPI(BaseScoreAPI, PyTorchAccuracyAPI, PyTorchLossAPI):
    pass
