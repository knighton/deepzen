from ....base.transform.shape import BaseShapeAPI
from .pad import MXNetPadAPI
from .pool import MXNetPoolAPI


class MXNetShapeAPI(BaseShapeAPI, MXNetPadAPI, MXNetPoolAPI):
    pass
