from mxnet.ndarray import NDArray

from ...base.core.data_type import BaseDataTypeAPI


class MXNetDataTypeAPI(BaseDataTypeAPI):
    def __init__(self, dtypes, home_floatx):
        BaseDataTypeAPI.__init__(self, dtypes, home_floatx)

    def dtype(self, x=None):
        if isinstance(x, NDArray):
            x = x.dtype.__name__
        else:
            x = self.get_dtype(x)
        return x

    def floatx(self, x=None):
        if isinstance(x, NDArray):
            x = x.dtype.__name__
            assert x in self._floatxs
        else:
            x = self.get_floatx(x)
        return x
