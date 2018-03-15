from torch import _TensorBase
from torch.autograd import Variable

from ...base.core.data_type import BaseDataTypeAPI


class PyTorchDataTypeAPI(BaseDataTypeAPI):
    def __init__(self, tensor2dtype, home_floatx):
        dtypes = set()
        for dtype in tensor2dtype.values():
            dtypes.add(dtype)
        BaseDataTypeAPI.__init__(self, dtypes, home_floatx)
        self._tensor2dtype = tensor2dtype

    def dtype(self, x=None):
        if isinstance(x, _TensorBase):
            x = self._tensor2dtype[x.type()]
        elif isinstance(x, Variable):
            x = self._tensor2dtype[x.data.type()]
        else:
            x = self.get_dtype(x)
        return x

    def floatx(self, x=None):
        if isinstance(x, _TensorBase):
            x = self._tensor2dtype[x.type()]
            assert x in self._floatxs
        elif isinstance(x, Variable):
            x = self._tensor2dtype[x.data.type()]
            assert x in self._floatxs
        else:
            x = self.get_floatx(x)
        return x
