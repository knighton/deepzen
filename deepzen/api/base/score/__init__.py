from .accuracy import BaseAccuracyAPI
from .loss import BaseLossAPI


class BaseScoreAPI(BaseAccuracyAPI, BaseLossAPI):
    pass
