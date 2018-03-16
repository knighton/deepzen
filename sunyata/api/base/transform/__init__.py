from .activ import BaseActivAPI
from .dot import BaseDotAPI
from .shape import BaseShapeAPI


class BaseTransformAPI(BaseActivAPI, BaseDotAPI, BaseShapeAPI):
    pass
