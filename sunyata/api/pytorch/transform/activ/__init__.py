from ....base.transform.activ import BaseActivAPI
from .softmax import PyTorchSoftmaxAPI


class PyTorchActivAPI(BaseActivAPI, PyTorchSoftmaxAPI):
    pass
