from ...base.transform import BaseTransformAPI
from .activ import MXNetActivAPI
from .dot import MXNetDotAPI


class MXNetTransformAPI(BaseTransformAPI, MXNetActivAPI, MXNetDotAPI):
    pass
