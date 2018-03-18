from ...base.transform import BaseTransformAPI
from .activ import MXNetActivAPI
from .arch import MXNetArchAPI
from .dot import MXNetDotAPI


class MXNetTransformAPI(BaseTransformAPI, MXNetActivAPI, MXNetArchAPI,
                        MXNetDotAPI):
    pass
