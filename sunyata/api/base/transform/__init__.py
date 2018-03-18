from .activ import BaseActivAPI
from .dot import BaseDotAPI
from .noise import BaseNoiseAPI
from .shape import BaseShapeAPI


class BaseTransformAPI(BaseActivAPI, BaseDotAPI, BaseNoiseAPI, BaseShapeAPI):
    pass
