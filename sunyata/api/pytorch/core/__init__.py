import importlib
import torch

from .cast import PyTorchCastAPI
from .data_type import PyTorchDataTypeAPI
from .device import PyTorchDeviceAPI


class PyTorchCoreAPI(PyTorchCastAPI, PyTorchDataTypeAPI, PyTorchDeviceAPI):
    def _init_api_pytorch_core(self, floatx='float32', device=None):
        config = """
            uint8    torch.ByteTensor    torch.cuda.ByteTensor
            int8     torch.CharTensor    torch.cuda.CharTensor
            int16    torch.ShortTensor   torch.cuda.ShortTensor
            int32    torch.IntTensor     torch.cuda.IntTensor
            int64    torch.LongTensor    torch.cuda.LongTensor
            float16  torch.HalfTensor    torch.cuda.HalfTensor
            float32  torch.FloatTensor   torch.cuda.FloatTensor
            float64  torch.DoubleTensor  torch.cuda.DoubleTensor
        """

        tensor2dtype = {}
        dtype_xpu2tensor = {}
        for line in config.strip().split('\n'):
            dtype, cpu, gpu = line.split()
            tensor2dtype[cpu] = dtype
            tensor2dtype[gpu] = dtype
            for xpu, path in [('cpu', cpu), ('gpu', gpu)]:
                x = path.rindex('.')
                module_name = path[:x]
                class_name = path[x + 1:]
                module = importlib.import_module(module_name)
                klass = getattr(module, class_name)
                dtype_xpu2tensor[(dtype, xpu)] = klass

        num_gpus = torch.cuda.device_count()

        self._init_api_pytorch_core_data_type(tensor2dtype, floatx)
        self._init_api_pytorch_core_device(num_gpus, device)
        self._init_api_pytorch_core_cast(dtype_xpu2tensor)
