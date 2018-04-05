from .activ import BaseActivAPI
from .arch import BaseArchAPI
from .dot import BaseDotAPI
from .merge import BaseMergeAPI
from .norm import BaseNormAPI
from .shape import BaseShapeAPI


class BaseTransformAPI(BaseActivAPI, BaseArchAPI, BaseDotAPI, BaseMergeAPI,
                       BaseNormAPI, BaseShapeAPI):
    pass
