import importlib
import torch

from ...base.core import BaseCoreAPI
from .cast import PyTorchCastAPI
from .data_type import PyTorchDataTypeAPI
from .device import PyTorchDeviceAPI
from .epsilon import PyTorchEpsilonAPI
from .logic import PyTorchLogicAPI
from .map import PyTorchMapAPI
from .reduce import PyTorchReduceAPI
from .reshape import PyTorchReshapeAPI


class PyTorchCoreAPI(BaseCoreAPI, PyTorchCastAPI, PyTorchDataTypeAPI,
                     PyTorchDeviceAPI, PyTorchEpsilonAPI, PyTorchLogicAPI,
                     PyTorchMapAPI, PyTorchReduceAPI, PyTorchReshapeAPI):
    def __init__(self, floatx='float32', device=None, epsilon=1e-5):
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

        BaseCoreAPI.__init__(self)
        PyTorchCastAPI.__init__(self, dtype_xpu2tensor)
        PyTorchDataTypeAPI.__init__(self, tensor2dtype, floatx)
        PyTorchDeviceAPI.__init__(self, num_gpus, device)
        PyTorchEpsilonAPI.__init__(self, epsilon)
        PyTorchLogicAPI.__init__(self)
        PyTorchMapAPI.__init__(self)
        PyTorchReduceAPI.__init__(self)
        PyTorchReshapeAPI.__init__(self)
