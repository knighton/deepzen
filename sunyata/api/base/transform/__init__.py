from .activ import BaseActivAPI
from .dot import BaseDotAPI
from .noise import BaseNoiseAPI
from .norm import BaseNormAPI
from .shape import BaseShapeAPI


class BaseTransformAPI(BaseActivAPI, BaseDotAPI, BaseNoiseAPI, BaseNormAPI,
                       BaseShapeAPI):
    pass
