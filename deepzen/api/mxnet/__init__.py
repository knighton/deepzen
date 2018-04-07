from .core import MXNetCoreAPI
from .layer import MXNetTransformAPI
from .meter import MXNetMeterAPI


class MXNetAPI(MXNetCoreAPI, MXNetMeterAPI, MXNetTransformAPI):
    def __init__(self, floatx='float32', device=None, epsilon=1e-5):
        MXNetCoreAPI.__init__(self, floatx, device, epsilon)
