from .alpha_dropout import BaseAlphaDropoutAPI
from .batch_norm import BaseBatchNormAPI
from .dropout import BaseDropoutAPI
from .gaussian_noise import BaseGaussianNoiseAPI
from .gaussian_dropout import BaseGaussianDropoutAPI


class BaseNormAPI(BaseAlphaDropoutAPI, BaseBatchNormAPI, BaseDropoutAPI,
                  BaseGaussianNoiseAPI, BaseGaussianDropoutAPI):
    pass
