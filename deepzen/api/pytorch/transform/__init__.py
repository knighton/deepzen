from ...base.transform import BaseTransformAPI
from .activ import PyTorchActivAPI
from .arch import PyTorchArchAPI
from .dot import PyTorchDotAPI
from .shape import PyTorchShapeAPI


class PyTorchTransformAPI(BaseTransformAPI, PyTorchActivAPI, PyTorchArchAPI,
                          PyTorchDotAPI, PyTorchShapeAPI):
    pass
