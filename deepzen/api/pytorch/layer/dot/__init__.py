from ....base.layer.dot import BaseDotAPI
from .conv import PyTorchConvAPI
from .dense import PyTorchDenseAPI


class PyTorchDotAPI(BaseDotAPI, PyTorchConvAPI, PyTorchDenseAPI):
    pass
