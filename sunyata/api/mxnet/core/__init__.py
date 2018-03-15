import subprocess

from ...base.core import BaseCoreAPI
from .cast import MXNetCastAPI
from .data_type import MXNetDataTypeAPI
from .device import MXNetDeviceAPI
from .epsilon import MXNetEpsilonAPI
from .logic import MXNetLogicAPI
from .map import MXNetMapAPI
from .reduce import MXNetReduceAPI
from .reshape import MXNetReshapeAPI
from .variable import MXNetVariableAPI


class MXNetCoreAPI(BaseCoreAPI, MXNetCastAPI, MXNetDataTypeAPI, MXNetDeviceAPI,
                   MXNetEpsilonAPI, MXNetLogicAPI, MXNetMapAPI, MXNetReduceAPI,
                   MXNetReshapeAPI, MXNetVariableAPI):
    def _discover_gpus(self):
        cmd = 'nvidia-smi -L'
        try:
            result = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
            lines = result.stdout.decode('unicode-escape')
            return len(lines)
        except:
            return 0

    def __init__(self, floatx='float32', device=None, epsilon=1e-5):
        config = """
            uint8  uint16  uint32  uint64
             int8   int16   int32   int64
                  float16 float32 float64
        """

        dtypes = set(config.split())

        num_gpus = self._discover_gpus()

        BaseCoreAPI.__init__(self)
        MXNetCastAPI.__init__(self)
        MXNetDataTypeAPI.__init__(self, dtypes, floatx)
        MXNetDeviceAPI.__init__(self, num_gpus, device)
        MXNetEpsilonAPI.__init__(self, epsilon)
        MXNetLogicAPI.__init__(self)
        MXNetMapAPI.__init__(self)
        MXNetReduceAPI.__init__(self)
        MXNetReshapeAPI.__init__(self)
        MXNetVariableAPI.__init__(self)
