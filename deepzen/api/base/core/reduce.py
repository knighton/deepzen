class BaseReduceAPI(object):
    """
    Reductions.
    """

    def argmin(self, axis=-1):
        """
        Argmin.

        in:
            tensor              x         Input tensor.
            {None, int, shape}  axis      Axes to reduce over.
            bool                keepdims  Whether to keep reduced dimensions.

        out:
            {scalar, tensor}    y         Argmins.
        """
        raise NotImplementedError

    def argmax(self, axis=-1):
        """
        Argmax.

        in:
            tensor              x         Input tensor.
            {None, int, shape}  axis      Axes to reduce over.
            bool                keepdims  Whether to keep reduced dimensions.

        out:
            {scalar, tensor}    y         Argmaxes.
        """
        raise NotImplementedError

    def min(self, x, axis=None, keepdims=False):
        """
        Minmum.

        in:
            tensor              x         Input tensor.
            {None, int, shape}  axis      Axes to reduce over.
            bool                keepdims  Whether to keep reduced dimensions.

        out:
            {scalar, tensor}    y         Minima.
        """
        raise NotImplementedError

    def max(self, x, axis=None, keepdims=False):
        """
        Maximum.

        in:
            tensor              x         Input tensor.
            {None, int, shape}  axis      Axes to reduce over.
            bool                keepdims  Whether to keep reduced dimensions.

        out:
            {scalar, tensor}    y         Maxima.
        """
        raise NotImplementedError

    def mean(self, x, axis=None, keepdims=False):
        """
        Mean.

        in:
            tensor              x         Input tensor.
            {None, int, shape}  axis      Axes to reduce over.
            bool                keepdims  Whether to keep reduced dimensions.

        out:
            {scalar, tensor}    y         Means.
        """
        old_size = self.size(x)
        x = self.sum(x, axis, keepdims)
        ratio = old_size / self.size(x)
        return x / ratio

    def sum(self, x, axis=None, keepdims=False):
        """
        Sum.

        in:
            tensor              x         Input tensor.
            {None, int, shape}  axis      Axes to reduce over.
            bool                keepdims  Whether to keep reduced dimensions.

        out:
            {scalar, tensor}    y         Sums.
        """
        raise NotImplementedError

    def prod(self, x, axis=None, keepdims=False):
        """
        Product.

        in:
            tensor              x         Input tensor.
            {None, int, shape}  axis      Axes to reduce over.
            bool                keepdims  Whether to keep reduced dimensions.

        out:
            {scalar, tensor}    y         Products.
        """
        raise NotImplementedError

    def var(self, x, axis=None, keepdims=False):
        """
        Variance.

        in:
            tensor              x         Input tensor.
            {None, int, shape}  axis      Axes to reduce over.
            bool                keepdims  Whether to keep reduced dimensions.

        out:
            {scalar, tensor}    y         Variances.
        """
        mean = self.mean(x, axis, keepdims)
        var = self.square(x - mean)
        return self.mean(var, axis, keepdims)

    def std(self, x, axis=None, keepdims=False):
        """
        Standard deviation.

        in:
            tensor              x         Input tensor.
            {None, int, shape}  axis      Axes to reduce over.
            bool                keepdims  Whether to keep reduced dimensions.

        out:
            {scalar, tensor}    y         Standard deviations.
        """
        return self.sqrt(self.var(x, axis, keepdims))

    def moments(self, x, axis=None, keepdims=False):
        """
        First and second moments (mean and variance).

        in:
            tensor              x         Input tensor.
            {None, int, shape}  axis      Axes to reduce over.
            bool                keepdims  Whether to keep reduced dimensions.

        out:
            {scalar, tensor}    means     Means.
            {scalar, tensor}    vars      Variances.
        """
        shift = self.mean(x, axis, True)
        shifted = x - shift
        shifted_mean = self.mean(shifted, axis, True)
        var_mean = self.mean(self.square(shifted), axis, True)
        var = var_mean - self.square(shifted_mean)
        mean = shifted_mean + shift
        return mean, var

    def any(self, x, axis=None, keepdims=False, dtype=None):
        """
        Any (whether any value is nonzero).

        in:
            tensor              x         Input tensor.
            {None, int, shape}  axis      Axes to reduce over.
            bool                keepdims  Whether to keep reduced dimensions.
            str                 dtype     Output dtype.

        out:
            {scalar, tensor}    y         Anys.
        """
        nonneg = self.abs(x)
        minima = self.min(nonneg, axis, keepdims)
        return self.less(0, minima, dtype)

    def all(self, x, axis=None, keepdims=False, dtype=None):
        """
        All (whether all values are nonzero).

        in:
            tensor              x         Input tensor.
            {None, int, shape}  axis      Axes to reduce over.
            bool                keepdims  Whether to keep reduced dimensions.
            str                 dtype     Output dtype.

        out:
            {scalar, tensor}    y         Alls.
        """
        nonneg = self.abs(x)
        nonzero = self.less(0, nonneg, 'int64')
        sums = self.sum(nonzero, axis, keepdims)
        return self.less(0, sums, dtype)
