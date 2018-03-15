from ..base import API
from .core import PyTorchCoreAPI
from .metric import PyTorchMetricAPI
from .transform import PyTorchTransformAPI


class PyTorchAPI(API):
    def __init__(self):
        self._api(PyTorchCoreAPI())
        self._api(PyTorchMetricAPI())
        self._api(PyTorchTransformAPI())
