from ....base.layer.activ import BaseActivAPI
from .map import PyTorchMapAPI
from .relative import PyTorchRelativeAPI


class PyTorchActivAPI(BaseActivAPI, PyTorchMapAPI, PyTorchRelativeAPI):
    pass
