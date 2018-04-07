from .global_pool import BaseGlobalPoolAPI
from .pad import BasePadAPI
from .pool import BasePoolAPI
from .reshape import BaseReshapeAPI
from .upsample import BaseUpsampleAPI


class BaseShapeAPI(BaseGlobalPoolAPI, BasePadAPI, BasePoolAPI, BaseReshapeAPI,
                   BaseUpsampleAPI):
    pass
