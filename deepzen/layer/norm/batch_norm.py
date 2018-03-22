from ... import api as Z
from ...init import unpack_initializer
from ..base.layer import Layer
from ..base.spec import Spec


class BaseBatchNormLayer(Layer):
    def __init__(self, x_sig):
        """
        BaseBatchNormLayer init.

        @params
            Signature  x_sig  Input signature.
        """
        Layer.__init__(self, x_sig, x_sig)


class InstanceBatchNormLayer(BaseBatchNormLayer):
    def __init__(self, x_sig, beta, gamma):
        """
        InstanceBatchNormLayer init.

        @params
            Signature   x_sig  Input signature.
            np.ndarray  beta   Beta weights.
            np.ndarray  gamma  Gamma weights.
        """
        BaseBatchNormLayer.__init__(self, x_sig)
        self._beta = self.param(beta)
        self._gamma = self.param(gamma)

    def forward(self, x, is_training):
        """
        Forward pass.

        @params
            tensor  x            Input tensor.
            bool    is_training  Whether in training mode.

        @returns
            tensor  y            Output tensor.
        """
        return Z.instance_batch_norm(x, self._beta, self._gamma)


class MovAvgBatchNormLayer(BaseBatchNormLayer):
    def __init__(self, x_sig, momentum, beta, gamma, mean, var):
        """
        MovAvgBatchNormLayer init.

        @params
            Signature     x_sig       Input signature.
            scalar        momentum    Momentum of mean and var moving averages.
            np.ndarray    beta        Beta weights.
            np.ndarray    gamma       Gamma weights.
            np.ndarray    mean        Mean moving average weights.
            np.ndarray    var         Var moving average weights.
        """
        BaseBatchNormLayer.__init__(self, x_sig)
        self._momentum = momentum
        self._beta = self.param(beta)
        self._gamma = self.param(gamma)
        self._mean = self.param(mean, learned=False)
        self._var = self.param(var, learned=False)

    def forward(self, x, is_training):
        """
        Forward pass.

        @params
            tensor  x            Input tensor.
            bool    is_training  Whether in training mode.

        @returns
            tensor  y            Output tensor.
        """
        return Z.mov_avg_batch_norm(x, is_training, self._momentum, self._beta,
                                    self._gamma, self._mean, self._var)


class BaseBatchNormSpec(Spec):
    def __init__(self, space):
        """
        BaseBatchNormSpec init.

        @params
            {None, int}  space  Optional x spatial ndim requirement.
        """
        Spec.__init__(self, space)

    @classmethod
    def _get_state_init_args(cls, axis, x_sig):
        if isinstance(axis, int):
            axes = [axis]
        elif isinstance(axis, (list, tuple)):
            axes = axis
        else:
            assert False
        state_shape = list(x_sig.batch_shape(1))
        for axis in axes:
            state_shape[1 + axis] = 1
        return tuple(state_shape), x_sig.dtype()


class InstanceBatchNormSpec(BaseBatchNormSpec):
    def __init__(self, axis=0, beta_init='zero', gamma_init='one', center=True,
                 scale=True, space=None):
        """
        InstanceBatchNormSpec init.

        The default sample axis to normalize over (the channels dim -- axis 0)
        probably shouldn't be changed unless you know what you're doing.

        @params
            {int, shape}  axis        Sample axis/axes to norm over.
            Initializer   beta_init   Beta weight initializer.
            Initializer   gamma_init  Gamma weight initializer.
            {None, int}   space       Optional x spatial ndim requirement.
        """
        BaseBatchNormSpec.__init__(self, space)
        self._axis = axis
        self._beta_init = unpack_initializer(beta_init)
        self._gamma_init = unpack_initializer(gamma_init)
        self._center = center
        self._scale = scale

    def checked_build(self, x_sig):
        """
        Create a layer from this spec.

        @params
            Signature             x_sig  Input signature.

        @returns
            MovAvgBatchNormLayer  layer  The specified layer.
        """
        args = self._get_state_init_args(self._axis, x_sig)
        if self._center:
            beta = self._beta_init(*args)
        else:
            beta = None
        if self._scale:
            gamma = self._gamma_init(*args)
        else:
            gamma = None
        return InstanceBatchNormLayer(x_sig, beta, gamma)


class MovAvgBatchNormSpec(BaseBatchNormSpec):
    def __init__(self, momentum=0.99, axis=0, beta_init='zero',
                 gamma_init='one', mean_init='zero', var_init='one',
                 center=True, scale=True, space=None):
        """
        MovAvgBatchNormSpec init.

        Set momentum to zero to get InstanceBatchNorm behavior, one for values
        fixed at initialization, or something in-between for normal behavior.

        The default sample axis to normalize over (the channels dim -- axis 0)
        probably shouldn't be changed unless you know what you're doing.

        @params
            scalar        momentum    Momentum of mean and var moving averages.
            {int, shape}  axis        Sample axis/axes to norm over.
            Initializer   beta_init   Beta weight initializer.
            Initializer   gamma_init  Gamma weight initializer.
            Initializer   mean_init   Mean moving average weight initializer.
            Initializer   var_init    Var moving average weight initializer.
            {None, int}   space       Optional x spatial ndim requirement.
        """
        BaseBatchNormSpec.__init__(self, space)
        assert 0 <= momentum <= 1
        self._momentum = momentum
        self._axis = axis
        self._beta_init = unpack_initializer(beta_init)
        self._gamma_init = unpack_initializer(gamma_init)
        self._mean_init = unpack_initializer(mean_init)
        self._var_init = unpack_initializer(var_init)
        self._center = center
        self._scale = scale

    def checked_build(self, x_sig):
        """
        Create a layer from this spec.

        @params
            Signature             x_sig  Input signature.

        @returns
            MovAvgBatchNormLayer  layer  The specified layer.
        """
        args = self._get_state_init_args(self._axis, x_sig)
        if self._center:
            beta = self._beta_init(*args)
        else:
            beta = None
        if self._scale:
            gamma = self._gamma_init(*args)
        else:
            gamma = None
        mean = self._mean_init(*args)
        var = self._var_init(*args)
        return MovAvgBatchNormLayer(x_sig, self._momentum, beta, gamma, mean,
                                    var)
