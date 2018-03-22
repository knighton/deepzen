from .activ import BaseActivAPI
from .arch import BaseArchAPI
from .dot import BaseDotAPI
from .noise import BaseNoiseAPI
from .norm import BaseNormAPI
from .shape import BaseShapeAPI


class BaseTransformAPI(BaseActivAPI, BaseArchAPI, BaseDotAPI, BaseNoiseAPI,
                       BaseNormAPI, BaseShapeAPI):
    pass
