from ....base.transform.activ import BaseActivAPI
from .map import MXNetMapAPI
from .relative import MXNetRelativeAPI


class MXNetActivAPI(BaseActivAPI, MXNetMapAPI, MXNetRelativeAPI):
    pass
