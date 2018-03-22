from .core import MXNetCoreAPI
from .score import MXNetScoreAPI
from .transform import MXNetTransformAPI


class MXNetAPI(MXNetCoreAPI, MXNetScoreAPI, MXNetTransformAPI):
    def __init__(self, floatx='float32', device=None, epsilon=1e-5):
        MXNetCoreAPI.__init__(self, floatx, device, epsilon)
