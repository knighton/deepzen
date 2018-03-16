from .core import MXNetCoreAPI
from .metric import MXNetMetricAPI
from .transform import MXNetTransformAPI


class MXNetAPI(MXNetCoreAPI, MXNetMetricAPI, MXNetTransformAPI):
    def __init__(self, floatx='float32', device=None, epsilon=1e-5):
        MXNetCoreAPI.__init__(self, floatx, device, epsilon)
