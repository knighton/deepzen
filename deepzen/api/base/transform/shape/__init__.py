from .global_pool import BaseGlobalPoolAPI
from .pool import BasePoolAPI
from .reshape import BaseReshapeAPI


class BaseShapeAPI(BaseGlobalPoolAPI, BasePoolAPI, BaseReshapeAPI):
    pass
