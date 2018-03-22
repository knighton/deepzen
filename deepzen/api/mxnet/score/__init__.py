from ...base.score import BaseScoreAPI
from .accuracy import MXNetAccuracyAPI
from .loss import MXNetLossAPI


class MXNetScoreAPI(BaseScoreAPI, MXNetAccuracyAPI, MXNetLossAPI):
    pass
