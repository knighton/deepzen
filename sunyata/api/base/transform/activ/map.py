import numpy as np


class BaseArctanAPI(object):
    def arctan_dx(self, x):
        return 1 / (self.square(x) + 1)


class BaseBentIdentityAPI(object):
    def bent_identity(self, x):
        return (self.sqrt(self.square(x) + 1) - 1) / 2 + x

    def bent_identity_dx(self, x):
        return x / (2 * self.sqrt(self.square(x) + 1)) + 1


class BaseClipAPI(object):
    def clip_dx(self, x, min=-np.inf, max=np.inf):
        return self.less(min, x) * self.less(x, max)


class BaseELUAPI(object):
    def elu(self, x, alpha=1):
        return self.where(self.less(x, 0), alpha * self.expm1(x), x)

    def elu_dx(self, x, alpha=1):
        return self.where(self.less(x, 0), self.elu(x, alpha) + alpha, 1)


class BaseHardShrinkAPI(object):
    def hard_shrink(self, x, lam=0.5):
        return (self.less(x, -lam) + self.less(lam, x)) * x

    def hard_shrink_dx(self, x, lam=0.5):
        return self.less(x, -lam) + self.less(lam, x)


class BaseHardSigmoidAPI(object):
    def hard_sigmoid(self, x):
        return self.clip(0.2 * x + 0.5, 0, 1)

    def hard_sigmoid_dx(self, x):
        return 0.2 * self.less(-2.5, x) * self.less(x, 2.5)


class BaseHardTanhAPI(object):
    def hard_tanh(self, x):
        return self.clip(x, -1, 1)

    def hard_tanh_dx(self, x):
        return self.less(-1, x) * self.less(x, 1)


class BaseIdentityAPI(object):
    def identity(self, x):
        return x

    def identity_dx(self, x):
        return 0 * x + 1


class BaseLeakyReLUAPI(object):
    def leaky_relu(self, x, alpha=0.1):
        x = self.relu(x)
        if alpha != 0:
            x -= alpha * self.relu(-x)
        return x

    def leaky_relu_dx(self, x, alpha=0.1):
        return self.where(self.less(x, 0), alpha, 1)


class BaseLogSigmoidAPI(object):
    def log_sigmoid(self, x):
        return self.log(self.sigmoid(x))

    def log_sigmoid_dx(self, x):
        return 1 / (self.exp(x) + 1)


class BaseReLUAPI(object):
    def relu(self, x):
        return self.clip(x, 0, np.inf)

    def relu_dx(self, x):
        return self.clip_dx(x, 0, np.inf)


class BaseReLU6API(object):
    def relu6(self, x):
        return self.clip(x, 0, 6)

    def relu6_dx(self, x):
        return self.clip_dx(x, 0, 6)


class BaseSELUAPI(object):
    SELU_ALPHA = 1.6732632423543772848170429916717
    SELU_SCALE = 1.0507009873554804934193349852946

    def selu(self, x):
        return self.SELU_SCALE * self.elu(x, self.SELU_ALPHA)

    def selu_dx(self, x):
        return self.SELU_SCALE * self.elu_dx(x, self.SELU_ALPHA)


class BaseSigmoidAPI(object):
    def sigmoid(self, x):
        return 1 / (self.exp(-x) + 1)

    def sigmoid_dx(self, x):
        return self.exp(x) / self.square(self.exp(x) + 1)


class BaseSoftExponentialAPI(object):
    def soft_exponential(self, x, alpha=0.25):
        if alpha < 0:
            x = -self.log(1 - alpha * (x + alpha)) / alpha
        elif alpha == 0:
            pass
        else:
            x = self.expm1(alpha * x) / alpha + alpha
        return x

    def soft_exponential_dx(self, x, alpha=0.25):
        if alpha < 0:
            x = 1 / (1 - alpha * (x + alpha))
        elif alpha == 0:
            x = 1
        else:
            x = self.exp(alpha * x)
        return x


class BaseSoftplusAPI(object):
    def softplus(self, x, beta=1, threshold=20):
        curve = 1 / beta * self.log1p(self.exp(beta * x))
        return self.where(self.less(x, threshold), curve, x)

    def softplus_dx(self, x, beta=1, threshold=20):
        e = self.exp(beta * x)
        curve = e / (e + 1)
        return self.where(self.less(x, threshold), curve, 1)


class BaseSoftshrinkAPI(object):
    def softshrink(self, x, lam=0.5):
        return self.sign(x) * self.maximum(self.abs(x) - lam, 0)

    def softshrink_dx(self, x, lam=0.5):
        return self.less(x, -lam) + self.less(lam, x)


class BaseSoftsignAPI(object):
    def softsign(self, x):
        return x / (self.abs(x) + 1)

    def softsign_dx(self, x):
        return 1 / self.square(self.abs(x) + 1)


class BaseSwishAPI(object):
    def swish(self, x, beta=1):
        return x * self.sigmoid(beta * x)

    def swish_dx(self, x, beta=1):
        f_x = self.swish(x)
        return beta * f_x + self.sigmoid(beta * x) * (1 - beta * f_x)


class BaseTanhAPI(object):
    def tanh_dx(self, x):
        return 4 / self.square(self.exp(-x) + self.exp(x))


class BaseTanhShrinkAPI(object):
    def tanh_shrink(self, x):
        return x - self.tanh(x)

    def tanh_shrink_dx(self, x):
        num = self.expm1(2 * x)
        denom = self.exp(2 * x) + 1
        return self.square(num) / self.square(denom)


class BaseThresholdAPI(object):
    def threshold(self, x):
        return self.clip(x, -1, np.inf)

    def threshold_dx(self, x):
        return self.clip_dx(x, -1, np.inf)


class BaseMapAPI(BaseArctanAPI, BaseBentIdentityAPI, BaseELUAPI,
                 BaseHardShrinkAPI, BaseHardSigmoidAPI, BaseHardTanhAPI,
                 BaseIdentityAPI, BaseLeakyReLUAPI, BaseLogSigmoidAPI,
                 BaseReLUAPI, BaseSELUAPI, BaseSigmoidAPI,
                 BaseSoftExponentialAPI, BaseSoftplusAPI, BaseSoftshrinkAPI,
                 BaseSoftsignAPI, BaseSwishAPI, BaseTanhAPI, BaseTanhShrinkAPI):
    pass
