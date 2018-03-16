from .relu import BaseReLUAPI
from .softmax import BaseSoftmaxAPI


class BaseActivAPI(BaseReLUAPI, BaseSoftmaxAPI):
    pass
