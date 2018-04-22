from .channel_pool import BaseChannelPoolAPI
from .global_pool import BaseGlobalPoolAPI
from .pad import BasePadAPI
from .pool import BasePoolAPI
from .reshape import BaseReshapeAPI
from .upsample import BaseUpsampleAPI


class BaseShapeAPI(BaseChannelPoolAPI, BaseGlobalPoolAPI, BasePadAPI,
                   BasePoolAPI, BaseReshapeAPI, BaseUpsampleAPI):
    pass
