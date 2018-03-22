from ...base.transform import BaseTransformAPI
from .activ import PyTorchActivAPI
from .arch import PyTorchArchAPI
from .dot import PyTorchDotAPI


class PyTorchTransformAPI(BaseTransformAPI, PyTorchActivAPI, PyTorchArchAPI,
                          PyTorchDotAPI):
    pass
