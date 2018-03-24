from .global_pool import BaseGlobalPoolAPI
from .pad import BasePadAPI
from .pool import BasePoolAPI
from .reshape import BaseReshapeAPI


class BaseShapeAPI(BaseGlobalPoolAPI, BasePadAPI, BasePoolAPI, BaseReshapeAPI):
    pass
