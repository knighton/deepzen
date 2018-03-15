from .cast import MXNetCastAPI
from .data_type import MXNetDataTypeAPI
from .device import MXNetDeviceAPI


class MXNetCoreAPI(MXNetCastAPI, MXNetDataTypeAPI, MXNetDeviceAPI):
    def _discover_gpus(self):
        cmd = 'nvidia-smi', '-L'
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE)
            lines = result.stdout.decode('unicode-escape')
            return len(lines)
        except:
            return 0

    def _init_api_mxnet_core(self, floatx='float32', device=None):
        config = """
            uint8  uint16  uint32  uint64
             int8   int16   int32   int64
                  float16 float32 float64
        """

        dtypes = set(config.split())

        num_gpus = self._discover_gpus()

        self._init_api_mxnet_core_data_type(dtypes, floatx)
        self._init_api_mxnet_core_device(num_gpus, device)
        self._init_api_mxnet_core_cast()
