from ...base.transform import BaseTransformAPI
from .activ import PyTorchActivAPI
from .dot import PyTorchDotAPI


class PyTorchTransformAPI(BaseTransformAPI, PyTorchActivAPI, PyTorchDotAPI):
    pass
