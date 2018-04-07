from ...base.meter import BaseMeterAPI
from .accuracy import MXNetAccuracyAPI
from .loss import MXNetLossAPI


class MXNetMeterAPI(BaseMeterAPI, MXNetAccuracyAPI, MXNetLossAPI):
    pass
