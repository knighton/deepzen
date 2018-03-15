import torch

from ...base.core.cast import BaseCastAPI
from .data_type import PyTorchDataTypeAPI
from .device import PyTorchDeviceAPI


class PyTorchCastAPI(BaseCastAPI, PyTorchDataTypeAPI, PyTorchDeviceAPI):
    def _init_api_pytorch_core_cast(self, dtype_xpu2tensor):
        self._init_api_base_core_cast()
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
