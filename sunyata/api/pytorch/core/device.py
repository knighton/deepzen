from torch import _TensorBase
from torch.autograd import Variable

from ...base.core.device import BaseDeviceAPI


class PyTorchDeviceAPI(BaseDeviceAPI):
    def _init_pytorch_device_api(self, num_gpus, home_device=None):
        self._init_base_device_api(num_gpus, home_device)

    def device(self, x=None):
        if isinstance(x, (_TensorBase, Variable)):
            if x.is_cuda:
                device_id = 1 + x.get_device()
            else:
                device_id = 0
            device = self._devices[device_id]
        else:
            device = self.get_device(x)
        return device

    def gpu(self, x=None):
        if isinstance(x, (_TensorBase, Variable)):
            if x.is_cuda:
                gpu_id = x.get_device()
            else:
                assert False
            gpu = self._gpus[gpu_id]
        else:
            gpu = self.get_gpu(x)
        return gpu
