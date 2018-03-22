import mxnet as mx
import numpy as np

from ...base.core.map import \
    BaseMapAPI, BaseClipAPI, BaseCumulativeAPI, BaseHyperbolicAPI, \
    BaseLogExpAPI, BasePowerAPI, BaseRoundAPI, BaseSignAPI, \
    BaseTrigonometricAPI


class MXNetClipAPI(BaseClipAPI):
    def clip(self, x, min=-np.inf, max=np.inf):
        return mx.nd.clip(x, min, max)


class MXNetCumulativeAPI(BaseCumulativeAPI):
    def cumsum(self, x, axis):
        raise NotImplementedError

    def cumprod(self, x, axis):
        raise NotImplementedError


class MXNetHyperbolicAPI(BaseHyperbolicAPI):
    def sinh(self, x):
        return mx.nd.sinh(x)

    def cosh(self, x):
        return mx.nd.cosh(x)

    def tanh(self, x):
        return mx.nd.tanh(x)

    def arcsinh(self, x):
        return mx.nd.arcsinh(x)

    def arccosh(self, x):
        return mx.nd.arccosh(x)

    def arctanh(self, x):
        return mx.nd.arctanh(x)


class MXNetLogExpAPI(BaseLogExpAPI):
    def __init__(self):
        BaseLogExpAPI.__init__(self)

    def exp(self, x):
        return mx.nd.exp(x)

    def expm1(self, x):
        return mx.nd.expm1(x)

    def log(self, x):
        return mx.nd.log(x)

    def log2(self, x):
        return mx.nd.log2(x)

    def log10(self, x):
        return mx.nd.log10(x)

    def log1p(self, x):
        return mx.nd.log1p(x)


class MXNetPowerAPI(BasePowerAPI):
    def __init__(self):
        BasePowerAPI.__init__(self)

    def pow(self, x, a):
        return mx.nd.power(x, a)

    def rsqrt(self, x):
        return mx.nd.rsqrt(x)

    def sqrt(self, x):
        return mx.nd.sqrt(x)

    def square(self, x):
        return mx.nd.square(x)


class MXNetRoundAPI(BaseRoundAPI):
    def __init__(self):
        BaseRoundAPI.__init__(self)

    def ceil(self, x):
        return mx.nd.ceil(x)

    def floor(self, x):
        return mx.nd.floor(x)

    def round(self, x):
        return mx.nd.round(x)

    def trunc(self, x):
        return mx.nd.trunc(x)


class MXNetSignAPI(BaseSignAPI):
    def __init__(self):
        BaseSignAPI.__init__(self)

    def abs(self, x):
        return mx.nd.abs(x)

    def neg(self, x):
        return mx.nd.negative(x)

    def sign(self, x):
        return mx.nd.sign(x)


class MXNetTrigonometricAPI(BaseTrigonometricAPI):
    def __init__(self):
        BaseTrigonometricAPI.__init__(self)

    def sin(self, x):
        return mx.nd.sin(x)

    def cos(self, x):
        return mx.nd.cos(x)

    def tan(self, x):
        return mx.nd.tan(x)

    def arcsin(self, x):
        return mx.nd.arcsin(x)

    def arccos(self, x):
        return mx.nd.arccos(x)

    def arctan(self, x):
        return mx.nd.arctan(x)


class MXNetMapAPI(BaseMapAPI, MXNetClipAPI, MXNetCumulativeAPI,
                  MXNetHyperbolicAPI, MXNetLogExpAPI, MXNetPowerAPI,
                  MXNetRoundAPI, MXNetSignAPI, MXNetTrigonometricAPI):
    pass
