import numpy as np
from torch import Tensor
from torch.autograd import Variable

from ...base.core.type import BaseTypeAPI


class PyTorchTypeAPI(BaseTypeAPI):
    def numpy(self, x):
        if np.isscalar(x):
            x = np.array(x)
        elif isinstance(x, list):
            x = np.array(x)
        elif isinstance(x, np.ndarray):
            pass
        elif isinstance(x, Tensor):
            if x.is_cuda:
                x = x.detach().cpu().numpy()
            else:
                x = x.numpy()
        elif isinstance(x, Variable):
            if x.data.is_cuda:
                x = x.detach().data.cpu().numpy()
            else:
                x = x.detach().data.numpy()
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
        elif isinstance(x, Tensor):
            x = self.cast(x, dtype, device, True)
        elif isinstance(x, Variable):
            x = self.cast(x.data, dtype, device, True)
        else:
            assert False
        return x

    def constant(self, x, dtype=None, device=None):
        x = self.tensor(x, dtype, device)
        return Variable(x, requires_grad=False)

    def variable(self, x, dtype=None, device=None):
        x = self.tensor(x, dtype, device)
        return Variable(x, requires_grad=True)

    def copy(self, x):
        return x.clone()
