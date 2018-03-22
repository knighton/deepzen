from ...base.metric import BaseMetricAPI
from .accuracy import MXNetAccuracyAPI
from .loss import MXNetLossAPI


class MXNetMetricAPI(BaseMetricAPI, MXNetAccuracyAPI, MXNetLossAPI):
    pass
