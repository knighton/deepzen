from ....base.transform.shape import BaseShapeAPI
from .pad import MXNetPadAPI
from .pool import MXNetPoolAPI
from .upsample import MXNetUpsampleAPI


class MXNetShapeAPI(BaseShapeAPI, MXNetPadAPI, MXNetPoolAPI, MXNetUpsampleAPI):
    pass
