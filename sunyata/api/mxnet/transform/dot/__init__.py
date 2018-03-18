from ....base.transform.dot import BaseDotAPI
from .conv import MXNetConvAPI
from .dense import MXNetDenseAPI


class MXNetDotAPI(BaseDotAPI, MXNetConvAPI, MXNetDenseAPI):
    pass
