from ....base.transform.activ import BaseActivAPI
from .softmax import MXNetSoftmaxAPI


class MXNetActivAPI(BaseActivAPI, MXNetSoftmaxAPI):
    pass
