from ....base.transform.dot import BaseDotAPI
from .dense import PyTorchDenseAPI


class PyTorchDotAPI(BaseDotAPI, PyTorchDenseAPI):
    pass
