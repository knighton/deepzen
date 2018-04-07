from .core import PyTorchCoreAPI
from .meter import PyTorchMeterAPI
from .transform import PyTorchTransformAPI


class PyTorchAPI(PyTorchCoreAPI, PyTorchMeterAPI, PyTorchTransformAPI):
    def __init__(self, floatx='float32', device=None, epsilon=1e-5):
        PyTorchCoreAPI.__init__(self, floatx, device, epsilon)
