import mxnet as mx
import subprocess

from ...base.core.cast import BaseCastAPI
from .data_type import MXNetDataTypeAPI
from .device import MXNetDeviceAPI


class MXNetCastAPI(object):
    def _init_api_mxnet_core_cast(self):
        pass

    def cast(self, x, dtype=None, device=None, copy=False):
        # Get the input and output dtypes (None means don't change).
        from_dtype = self.dtype(x)
        if dtype is None:
            to_dtype = from_dtype
        else:
            to_dtype = self.get_dtype(dtype)

        # Cast if desired.
        if from_dtype != to_dtype or copy:
            x = x.astype(to_dtype)
            copy = False

        # Get the input and output devices (None means don't change).
        from_device = self.device(x)
        if device is None:
            to_device = from_device
        else:
            to_device = self.get_device(device)

        # Move if desired.
        if from_device is not to_device:
            x = x.as_in_context(to_device.mx_ctx)
            copy = False

        # Copy if still needed.
        if copy:
            x = x.copy()

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

        # Create the tensor on the desired device.
        return mx.nd.array(x, to_device.mx_ctx, to_dtype)
