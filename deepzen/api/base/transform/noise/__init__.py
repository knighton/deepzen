from .alpha_dropout import BaseAlphaDropoutAPI
from .dropout import BaseDropoutAPI


class BaseNoiseAPI(BaseAlphaDropoutAPI, BaseDropoutAPI):
    pass
