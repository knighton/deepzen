from .core import MXNetCoreAPI
from .metric import MXNetMetricAPI
from .transform import MXNetTransformAPI


class MXNetAPI(MXNetCoreAPI, MXNetMetricAPI, MXNetTransformAPI):
    def __init__(self, floatx='float32', device=None):
        self._init_mxnet_core_api(floatx, device)
