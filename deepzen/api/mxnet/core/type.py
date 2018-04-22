import numpy as np
import mxnet as mx

from ...base.core.type import BaseTypeAPI


class MXNetTypeAPI(BaseTypeAPI):
    def numpy(self, x):
        if np.isscalar(x):
            x = np.array(x)
        elif isinstance(x, list):
            x = np.array(x)
        elif isinstance(x, np.ndarray):
            pass
        elif isinstance(x, mx.nd.NDArray):
            x = x.asnumpy()
        else:
            assert False
        return x

    def tensor(self, x, dtype=None, device=None):
        if np.isscalar(x):
            x = np.array(x)
            x = self.cast_numpy_to_tensor(x, dtype, device)
        elif isinstance(x, list):
            x = np.array(x)
            x = self.cast_numpy_to_tensor(x, dtype, device)
        elif isinstance(x, np.ndarray):
            x = self.cast_numpy_to_tensor(x, dtype, device)
        elif isinstance(x, mx.nd.NDArray):
            x = self.cast(x, dtype, device, True)
        else:
            assert False
        return x

    def constant(self, x, dtype=None, device=None):
        return self.tensor(x, dtype, device)

    def variable(self, x, dtype=None, device=None):
        x = self.tensor(x, dtype, device)
        x.attach_grad()
        return x

    def copy(self, x):
        return x.copy()
