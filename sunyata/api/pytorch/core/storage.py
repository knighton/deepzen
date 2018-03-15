import importlib
import torch

from ...base.core.storage import BaseStorageAPI
from .data_type import PyTorchDataTypeAPI
from .device import PyTorchDeviceAPI


class PyTorchStorageAPI(BaseStorageAPI, PyTorchDataTypeAPI, PyTorchDeviceAPI):
    def _init_pytorch_storage_api(self, floatx='float32', device=None):
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
        self._init_pytorch_data_type_api(tensor2dtype, floatx)

        num_gpus = torch.cuda.device_count()
        self._init_pytorch_device_api(num_gpus, device)

        self._dtype_xpu2tensor = dtype_xpu2tensor

    def cast(self, x, dtype=None, device=None, copy=False):
        # Get the input and output dtypes (None means don't change).
        from_dtype = self.dtype(x)
        if dtype is None:
            to_dtype = from_dtype
        else:
            to_dtype = self.get_dtype(dtype)

        # Get the input and output devices (None means don't change).
        from_device = self.device(x)
        if device is None:
            to_device = from_device
        else:
            to_device = self.get_device(device)

        # Get the output PyTorch tensor class.
        to_tensor_class = self._dtype_xpu2tensor[(to_dtype, to_device.type)]

        # Perform the cast and/or move.
        if from_device is to_device:
            if from_dtype != to_dtype or copy:
                x = x.type(to_tensor_class)
        else:
            if to_device.type == 'cpu':
                x = to_tensor_class(x)
            elif to_device.type == 'gpu':
                with torch.cuda.device(to_device.gpu_id()):
                    x = x.type(to_tensor_class)
            else:
                assert False
        return x

    def numpy_to_tensor(self, x, dtype=None, device=None):
        # Get the input and output dtypes (None means don't change).
        from_dtype = x.dtype.name
        if dtype is None:
            to_dtype = from_dtype
        else:
            to_dtype = self.get_dtype(dtype)

        # Get the input and output devices (None means current default device).
        to_device = self.get_device(device)

        # Get the output PyTorch tensor class.
        to_tensor_class = self._dtype_xpu2tensor[(to_dtype, to_device.type)]

        # Create the tensor on the desired device.
        if to_device.type == 'cpu':
            x = to_tensor_class(x)
        elif to_device.type == 'gpu':
            with torch.cuda.device(to_device.gpu_id()):
                x = x.type(to_tensor_class)
        else:
            assert False
        return x
