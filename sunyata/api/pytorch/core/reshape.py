import torch

from ...base.core.reshape import BaseReshapeAPI


class PyTorchReshapeAPI(BaseReshapeAPI):
    def reshape(self, x, shape):
        return x.view(shape)

    def expand_dims(self, x, axis):
        return x.unsqueeze(axis)

    def squeeze(self, x, axis=None):
        return x.squeeze(axis)

    def tile_axis(self, x, axis, repeat):
        repeats = [1] * self.ndim(x)
        repeats[axis] = repeat
        return x.repeat(*repeats)

    def tile(self, x, repeats):
        return x.repeat(*repeats)

    def transpose(self, x, axes):
        return x.permute(*axes)

    def split(self, x, axis):
        return x.split(1, axis)

    def concat(self, xx, axis):
        return torch.cat(xx, axis)

    def stack(self, xx, axis=0):
        return torch.stack(xx, axis)
