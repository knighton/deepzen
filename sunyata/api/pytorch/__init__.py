from .core.orig import PyTorchCoreAPI
from .metric import PyTorchMetricAPI
from .transform import PyTorchTransformAPI


class PyTorchAPI(PyTorchCoreAPI, PyTorchMetricAPI, PyTorchTransformAPI):
    def __init__(self, floatx='float32', device=None):
        self._init_api_pytorch_core(floatx, device)
