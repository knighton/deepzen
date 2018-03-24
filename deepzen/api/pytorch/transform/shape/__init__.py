from ....base.transform.shape import BaseShapeAPI
from .pool import PyTorchPoolAPI


class PyTorchShapeAPI(BaseShapeAPI, PyTorchPoolAPI):
    pass
