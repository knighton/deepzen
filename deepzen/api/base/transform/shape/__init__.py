from .pool import BasePoolAPI
from .reshape import BaseReshapeAPI


class BaseShapeAPI(BasePoolAPI, BaseReshapeAPI):
    pass
