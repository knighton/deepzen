import numpy as np

from ...base.core.map import \
    BaseMapAPI, BaseClipAPI, BaseCumulativeAPI, BaseHyperbolicAPI, \
    BaseLogExpAPI, BasePowerAPI, BaseRoundAPI, BaseSignAPI, \
    BaseTrigonometricAPI


class PyTorchClipAPI(BaseClipAPI):
    def clip(self, x, min=-np.inf, max=np.inf):
        return x.clamp(min, max)


class PyTorchCumulativeAPI(BaseCumulativeAPI):
    def cumsum(self, x, axis):
        return x.cumsum(axis)

    def cumprod(self, x, axis):
        return x.cumprod(axis)


class PyTorchHyperbolicAPI(BaseHyperbolicAPI):
    def sinh(self, x):
        return x.sinh()

    def cosh(self, x):
        return x.cosh()

    def tanh(self, x):
        return x.tanh()


class PyTorchLogExpAPI(BaseLogExpAPI):
    def exp(self, x):
        return x.exp()

    def expm1(self, x):
        return x.expm1()

    def log(self, x):
        return x.log()

    def log1p(self, x):
        return x.log1p()


class PyTorchPowerAPI(BasePowerAPI):
    def pow(self, x, a):
        return x.pow(a)

    def rsqrt(self, x):
        return x.rsqrt()

    def sqrt(self, x):
        return x.sqrt()


class PyTorchRoundAPI(BaseRoundAPI):
    def ceil(self, x):
        return x.ceil()

    def floor(self, x):
        return x.floor()

    def round(self, x):
        return x.round()

    def trunc(self, x):
        return x.trunc()


class PyTorchSignAPI(BaseSignAPI):
    def abs(self, x):
        return x.abs()

    def neg(self, x):
        return x.neg()

    def sign(self, x):
        return x.sign()


class PyTorchTrigonometricAPI(BaseTrigonometricAPI):
    def sin(self, x):
        return x.sin()

    def cos(self, x):
        return x.cos()

    def tan(self, x):
        return x.tan()

    def arcsin(self, x):
        return x.asin()

    def arccos(self, x):
        return x.acos()

    def arctan(self, x):
        return x.atan()


class PyTorchMapAPI(BaseMapAPI, PyTorchClipAPI, PyTorchCumulativeAPI,
                    PyTorchHyperbolicAPI, PyTorchLogExpAPI, PyTorchPowerAPI,
                    PyTorchRoundAPI, PyTorchSignAPI, PyTorchTrigonometricAPI):
    pass
