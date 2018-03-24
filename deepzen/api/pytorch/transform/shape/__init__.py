from ....base.transform.shape import BaseShapeAPI
from .pad import PyTorchPadAPI
from .pool import PyTorchPoolAPI


class PyTorchShapeAPI(BaseShapeAPI, PyTorchPadAPI, PyTorchPoolAPI):
    pass
