from ....base.transform.dot import BaseDotAPI
from .dense import MXNetDenseAPI


class MXNetDotAPI(BaseDotAPI, MXNetDenseAPI):
    pass
