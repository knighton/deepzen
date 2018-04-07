from .core import MXNetCoreAPI
from .meter import MXNetMeterAPI
from .transform import MXNetTransformAPI


class MXNetAPI(MXNetCoreAPI, MXNetMeterAPI, MXNetTransformAPI):
    def __init__(self, floatx='float32', device=None, epsilon=1e-5):
        MXNetCoreAPI.__init__(self, floatx, device, epsilon)
