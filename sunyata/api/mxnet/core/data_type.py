from mxnet.ndarray import NDArray

from ...base.core.data_type import BaseDataTypeAPI


class MXNetDataTypeAPI(BaseDataTypeAPI):
    def _init_api_mxnet_core_data_type(self, dtypes, home_floatx):
        self._init_api_base_core_data_type(dtypes, home_floatx)

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
