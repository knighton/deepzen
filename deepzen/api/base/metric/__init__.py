from .accuracy import BaseAccuracyAPI
from .loss import BaseLossAPI


class BaseMetricAPI(BaseAccuracyAPI, BaseLossAPI):
    pass
