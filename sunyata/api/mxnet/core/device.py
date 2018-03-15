import mxnet as mx
from mxnet.ndarray import NDArray

from ...base.core.device import BaseDeviceAPI


class MXNetDeviceAPI(BaseDeviceAPI):
    def _init_mxnet_device_api(self, num_gpus, home_device=None):
        self._init_base_device_api(num_gpus, home_device)
        for device in self._devices:
            if device.type == 'cpu':
                ctx = mx.cpu()
            elif device.type == 'gpu':
                ctx = mx.gpu(device.gpu_id)
            else:
                assert False
            device.mx_ctx = ctx

    def _ctx_to_device(self, ctx):
        xpu = ctx.device_type
        if xpu == 'cpu':
            device = self._cpu
        elif xpu == 'gpu':
            gpu_id = ctx.device_id
            device = self._gpus[gpu_id]
        else:
            assert False
        return device

    def _ctx_to_gpu(self, ctx):
        assert ctx.device_type == 'gpu'
        return self._gpus[ctx.device_id]

    def device(self, x=None):
        if isinstance(x, NDArray):
            device = self._ctx_to_device(x.context)
        else:
            device = self.get_device(x)
        return device

    def gpu(self, x=None):
        if isinstance(x, NDArray):
            device = self._ctx_to_gpu(x.context)
        else:
            device = self.get_device(x)
        return device
