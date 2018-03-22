class BaseLogicAPI(object):
    """
    Elementwise binary logical comparisons.
    """

    def minimum(self, a, b):
        """
        Elementwise minima of two tensors.

        in:
            tensor  a  First tensor.
            tensor  b  Second tensor.

        out:
            tensor  c  Elementwise minima.
        """
        raise NotImplementedError

    def maximum(self, a, b):
        """
        Elementwise maxima of two tensors.

        in:
            tensor  a  First tensor.
            tensor  b  Second tensor.

        out:
            tensor  c  Elementwise maxima.
        """
        raise NotImplementedError

    def where(self, cond, true, false):
        """
        Select values of two tensors based on a boolean condition tensor.

        in:
            tensor  cond  Condition tensor.
            tensor  a     First tensor.
            tensor  b     Second tensor.

        out:
            tensor  c     Elementwise a if cond else b.
        """
        if self.dtype(cond) != 'bool':
            cond = self.not_equal(cond, 0)
        return cond * true + (1 - cond) * false

    def less(self, a, b):
        """
        Elementwise a < b.

        in:
            tensor  a  First tensor.
            tensor  b  Second tensor.

        out:
            tensor  c  Elementwise a < b.
        """
        raise NotImplementedError

    def less_equal(self, a, b):
        """
        Elementwise a <= b.

        in:
            tensor  a  First tensor.
            tensor  b  Second tensor.

        out:
            tensor  c  Elementwise a <= b.
        """
        raise NotImplementedError

    def equal(self, a, b):
        """
        Elementwise a == b.

        in:
            tensor  a  First tensor.
            tensor  b  Second tensor.

        out:
            tensor  c  Elementwise a == b.
        """
        raise NotImplementedError

    def greater_equal(self, a, b):
        """
        Elementwise a >= b.

        in:
            tensor  a  First tensor.
            tensor  b  Second tensor.

        out:
            tensor  c  Elementwise a >= b.
        """
        raise NotImplementedError

    def greater(self, a, b):
        """
        Elementwise a > b.

        in:
            tensor  a  First tensor.
            tensor  b  Second tensor.

        out:
            tensor  c  Elementwise a > b.
        """
        raise NotImplementedError

    def not_equal(self, a, b):
        """
        Elementwise a != b.

        in:
            tensor  a  First tensor.
            tensor  b  Second tensor.

        out:
            tensor  c  Elementwise a != b.
        """
        raise NotImplementedError
