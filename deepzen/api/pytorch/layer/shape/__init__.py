from ....base.layer.shape import BaseShapeAPI
from .channel_pool import PyTorchChannelPoolAPI
from .pad import PyTorchPadAPI
from .pool import PyTorchPoolAPI
from .upsample import PyTorchUpsampleAPI


class PyTorchShapeAPI(BaseShapeAPI, PyTorchChannelPoolAPI, PyTorchPadAPI,
                      PyTorchPoolAPI, PyTorchUpsampleAPI):
    pass
