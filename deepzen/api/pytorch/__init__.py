from .core import PyTorchCoreAPI
from .layer import PyTorchTransformAPI
from .meter import PyTorchMeterAPI


class PyTorchAPI(PyTorchCoreAPI, PyTorchMeterAPI, PyTorchTransformAPI):
    def __init__(self, floatx='float32', device=None, epsilon=1e-5):
        PyTorchCoreAPI.__init__(self, floatx, device, epsilon)
