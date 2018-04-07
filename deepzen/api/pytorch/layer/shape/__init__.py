from ....base.layer.shape import BaseShapeAPI
from .pad import PyTorchPadAPI
from .pool import PyTorchPoolAPI
from .upsample import PyTorchUpsampleAPI


class PyTorchShapeAPI(BaseShapeAPI, PyTorchPadAPI, PyTorchPoolAPI,
                      PyTorchUpsampleAPI):
    pass
