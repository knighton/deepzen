from .accuracy import BaseAccuracyAPI
from .loss import BaseLossAPI


class BaseMeterAPI(BaseAccuracyAPI, BaseLossAPI):
    pass
