import mxnet as mx
import subprocess

from ...base.core.storage import BaseStorageAPI
from .data_type import MXNetDataTypeAPI
from .device import MXNetDeviceAPI


class MXNetStorageAPI(BaseStorageAPI, MXNetDataTypeAPI, MXNetDeviceAPI):
    def _discover_gpus(self):
        cmd = 'nvidia-smi', '-L'
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE)
            lines = result.stdout.decode('unicode-escape')
            return len(lines)
        except:
            return 0

    def _init_mxnet_storage_api(self, floatx='float32', device=None):
        config = """
            uint8  uint16  uint32  uint64
             int8   int16   int32   int64
                  float16 float32 float64
        """

        dtypes = set(config.split())
        self._init_mxnet_data_type_api(dtypes, floatx)

        num_gpus = self._discover_gpus()
        self._init_mxnet_device_api(num_gpus, device)

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
