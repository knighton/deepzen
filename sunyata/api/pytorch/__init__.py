from .core.orig import PyTorchCoreAPI
from .metric import PyTorchMetricAPI
from .transform import PyTorchTransformAPI


class PyTorchAPI(PyTorchCoreAPI, PyTorchMetricAPI, PyTorchTransformAPI):
    def __init__(self, floatx='float32', device=None, epsilon=1e-5):
        PyTorchCoreAPI.__init__(self, floatx, device, epsilon)
