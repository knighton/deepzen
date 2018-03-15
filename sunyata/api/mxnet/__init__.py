from ..base import API
from .core import MXNetCoreAPI
from .metric import MXNetMetricAPI
from .transform import MXNetTransformAPI


class MXNetAPI(API):
    def __init__(self):
        self._api(MXNetCoreAPI())
        self._api(MXNetMetricAPI())
        self._api(MXNetTransformAPI())
