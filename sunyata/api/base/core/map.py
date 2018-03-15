import numpy as np


class BaseClipAPI(object):
    """
    Clip API.
    """

    def clip(self, x, min=-np.inf, max=np.inf):
        """
        Clip x between min and max.

        in:
            tensor  x    Input tensor.
            scalar  min  Minimum value.
            scalar  max  Maximum value.

        out:
            tensor  y    Clipped x.
        """
        raise NotImplementedError


class BaseCumulativeAPI(object):
    """
    Accumulator API.
    """

    def cumsum(self, x, axis):
        """
        Cumulative sum.

        in:
            tensor  x     Input tensor.
            int     axis  Axis to accumlute across.

        out:
            tensor  y     Accumulated x.
        """
        raise NotImplementedError

    def cumprod(self, x, axis):
        """
        Cumulative product.

        in:
            tensor  x     Input tensor.
            int     axis  Axis to accumlute across.

        out:
            tensor  y     Accumulated x.
        """
        raise NotImplementedError


class BaseHyperbolicAPI(object):
    """
    Hyperbolic function API.
    """

    def sinh(self, x):
        """
        Hyperbolic sine.

        in:
            tensor  x  Input tensor.

        output:
            tensor  y  Hyperbolic sine of x.
        """
        return (self.exp(x) - self.exp(-x)) / 2

    def cosh(self, x):
        """
        Hyperbolic cosine.

        in:
            tensor  x  Input tensor.

        output:
            tensor  y  Hyperbolic cosine of x.
        """
        return (self.exp(x) + self.exp(-x)) / 2

    def tanh(self, x):
        """
        Hyperbolic tangent.

        in:
            tensor  x  Input tensor.

        output:
            tensor  y  Hyperbolic tangent of x.
        """
        e_x = self.exp(x)
        e_nx = self.exp(-x)
        return (e_x - e_nx) / (e_x + e_nx)

    def arcsinh(self, x):
        """
        Hyperbolic arcsine.

        in:
            tensor  x  Input tensor.

        output:
            tensor  y  Hyperbolic arcsine of x.
        """
        return self.log(x + self.sqrt(self.square(x) + 1))

    def arccosh(self, x):
        """
        Hyperbolic arccosine.

        in:
            tensor  x  Input tensor.

        output:
            tensor  y  Hyperbolic arccosine of x.
        """
        return self.log(x + self.sqrt(self.square(x) - 1))

    def arctanh(self, x):
        """
        Hyperbolic arctangent.

        in:
            tensor  x  Input tensor.

        output:
            tensor  y  Hyperbolic arctangent of x.
        """
        return 0.5 * self.log((1 + x) / (1 - x))


class BaseLogExpAPI(object):
    """
    Log and e^x API.
    """

    LOG_2 = np.log(2)
    LOG_10 = np.log(10)

    def exp(self, x):
        """
        Compute e^x.

        in:
            tensor  x  Input tensor.

        out:
            tensor  y  e^x.
        """
        raise NotImplementedError

    def expm1(self, x):
        """
        Compute e^x - 1.

        in:
            tensor  x  Input tensor.

        out:
            tensor  y  e^x - 1.
        """
        return self.exp(x) - 1

    def log(self, x):
        """
        Compute log_e(x).

        in:
            tensor  x  Input tensor.

        out:
            tensor  y  log_e(x).
        """
        raise NotImplementedError

    def log2(self, x):
        """
        Compute log_2(x).

        in:
            tensor  x  Input tensor.

        out:
            tensor  y  log_2(x).
        """
        return self.log(x) / self.LOG_2

    def log10(self, x):
        """
        Compute log_10(x).

        in:
            tensor  x  Input tensor.

        out:
            tensor  y  log_10(x).
        """
        return self.log(x) / self.LOG_10

    def log1p(self, x):
        """
        Compute log_e(x + 1).

        in:
            tensor  x  Input tensor.

        out:
            tensor  y  log_e(x + 1).
        """
        return self.log(x + 1)


class BasePowerAPI(object):
    """
    API for raising to a power.
    """

    def pow(self, x, power):
        """
        Power.

        in:
            tensor  x  Input tensor.

        out:
            tensor  y  x raised to the power.
        """
        raise NotImplementedError

    def rsqrt(self, x):
        """
        Reciprocal of the square root.

        in:
            tensor  x  Input tensor.

        out:
            tensor  y  Reciprocal of the square root of x.
        """
        return 1 / self.sqrt(x)

    def sqrt(self, x):
        """
        Square root.

        in:
            tensor  x  Input tensor.

        out:
            tensor  y  Square root of x.
        """
        return self.pow(x, 0.5)

    def square(self, x):
        """
        Square.

        in:
            tensor  x  Input tensor.

        out:
            tensor  y  Square of x.
        """
        return self.pow(x, 2)


class BaseRoundAPI(object):
    """
    Rounding API.
    """

    def ceil(self, x):
        """
        Round up to integer.

        in:
            tensor  x  Input tensor.

        out:
            tensor  y  Ceil of x.
        """
        raise NotImplementedError

    def floor(self, x):
        """
        Round down to integer.

        in:
            tensor  x  Input tensor.

        out:
            tensor  y  Floor of x.
        """
        raise NotImplementedError

    def round(self, x):
        """
        Round to nearest integer (up or down).

        in:
            tensor  x  Input tensor.

        out:
            tensor  y  Round of x.
        """
        return self.floor(x + 0.5)

    def trunc(self, x):
        """
        Round to the integer toward zero.

        in:
            tensor  x  Input tensor.

        out:
            tensor  y  Trunc of x.
        """
        is_pos = self.less(0, x)
        return self.where(is_pos, self.floor(x), -self.floor(-x))


class BaseSignAPI(object):
    """
    Sign (positive/negative) API.
    """

    def abs(self, x):
        """
        Absolute value.

        in:
            tensor  x  Input tensor.

        out:
            tensor  y  Absolute value of x.
        """
        return self.sign(x) * x

    def neg(self, x):
        """
        Negation (flip the signs).

        in:
            tensor  x  Input tensor.

        out:
            tensor  y  Negation of x.
        """
        return -1 * x

    def sign(self, x):
        """
        Get the signs of the scalars (+1 or -1).

        in:
            tensor  x  Input tensor.

        out:
            tensor  y  Signs (+1 or -1) of x's scalars.
        """
        return self.less(0, x) * 2 - 1


class BaseTrigonometricAPI(object):
    """
    Trigonometric function API.
    """

    def sin(self, x):
        """
        Sine.

        in:
            tensor  x  Input tensor.

        output:
            tensor  y  Sine of x.
        """
        raise NotImplementedError

    def cos(self, x):
        """
        Cosine.

        in:
            tensor  x  Input tensor.

        output:
            tensor  y  Cosine of x.
        """
        raise NotImplementedError

    def tan(self, x):
        """
        Tangent.

        in:
            tensor  x  Input tensor.

        output:
            tensor  y  Tangent of x.
        """
        raise NotImplementedError

    def arcsin(self, x):
        """
        Arcsine.

        in:
            tensor  x  Input tensor.

        output:
            tensor  y  Arcsine of x.
        """
        raise NotImplementedError

    def arccos(self, x):
        """
        Arccosine.

        in:
            tensor  x  Input tensor.

        output:
            tensor  y  Arccosine of x.
        """
        raise NotImplementedError

    def arctan(self, x):
        """
        Arctangent.

        in:
            tensor  x  Input tensor.

        output:
            tensor  y  Arctangent of x.
        """
        raise NotImplementedError


class BaseMapAPI(BaseClipAPI, BaseCumulativeAPI, BaseHyperbolicAPI,
                 BaseLogExpAPI, BasePowerAPI, BaseRoundAPI, BaseSignAPI,
                 BaseTrigonometricAPI):
    """
    Map operations (elementwise transformations).
    """
    pass
