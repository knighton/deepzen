from ...base.transform import BaseTransformAPI
from .activ import MXNetActivAPI
from .arch import MXNetArchAPI
from .dot import MXNetDotAPI
from .shape import MXNetShapeAPI


class MXNetTransformAPI(BaseTransformAPI, MXNetActivAPI, MXNetArchAPI,
                        MXNetDotAPI, MXNetShapeAPI):
    pass
