from torch.nn import functional as F

from ....base.layer.shape.channel_pool import \
    BaseChannelAvgPoolAPI, BaseChannelMaxPoolAPI, BaseChannelPoolAPI


class PyTorchChannelAvgPoolAPI(BaseChannelAvgPoolAPI):
    def channel_avg_pool(self, x, face, xsnd=None):
        if xsnd is None:
            xsnd = self.ndim(x) - 2
        if xsnd == 0:
            ret = self.channel_avg_pool0d(x, face)
        elif xsnd == 1:
            ret = self.channel_avg_pool1d(x, face)
        elif xsnd == 2:
            ret = self.channel_avg_pool2d(x, face)
        elif xsnd == 3:
            ret = self.channel_avg_pool3d(x, face)
        else:
            assert False
        return ret

    def channel_avg_pool0d(self, x, face):
        assert self.ndim(x) == 2
        in_channels = self.shape(x)[1]
        assert 0 <= face <= in_channels
        assert in_channels % face == 0
        x = self.expand_dims(x, 1)
        pool_face = face
        pool_stride = pool_face
        pool_padding = 0
        x = F.avg_pool1d(x, pool_face, pool_stride, pool_padding)
        return self.squeeze(x, 1)

    def channel_avg_pool1d(self, x, face):
        assert self.ndim(x) == 3
        in_channels = self.shape(x)[1]
        assert 0 <= face <= in_channels
        assert in_channels % face == 0
        x = self.expand_dims(x, 1)
        pool_face = face, 1, 1
        pool_stride = pool_face
        pool_padding = 0
        x = F.avg_pool2d(x, pool_face, pool_stride, pool_padding)
        return self.squeeze(x, 1)

    def channel_avg_pool2d(self, x, face):
        assert self.ndim(x) == 4
        in_channels = self.shape(x)[1]
        assert 0 <= face <= in_channels
        assert in_channels % face == 0
        x = self.expand_dims(x, 1)
        pool_face = face, 1, 1
        pool_stride = pool_face
        pool_padding = 0
        x = F.avg_pool3d(x, pool_face, pool_stride, pool_padding)
        return self.squeeze(x, 1)

    def channel_avg_pool3d(self, x, face):
        assert self.ndim(x) == 4
        in_channels = self.shape(x)[1]
        assert 0 <= face <= in_channels
        assert in_channels % face == 0
        pool_face = face, 1, 1
        pool_stride = pool_face
        pool_padding = 0
        xx = self.split(x, in_channels // face)
        for i, x in enumerate(xx):
            x = self.expand_dims(x, 1)
            x = F.avg_pool3d(x, pool_face, pool_stride, pool_padding)
            x = self.squeeze(x, 1)
            xx[i] = x
        return self.concat(xx, axis=1)


class PyTorchChannelMaxPoolAPI(BaseChannelMaxPoolAPI):
    def channel_max_pool(self, x, face, xsnd=None):
        if xsnd is None:
            xsnd = self.ndim(x) - 2
        if xsnd == 0:
            ret = self.channel_max_pool0d(x, face)
        elif xsnd == 1:
            ret = self.channel_max_pool1d(x, face)
        elif xsnd == 2:
            ret = self.channel_max_pool2d(x, face)
        elif xsnd == 3:
            ret = self.channel_max_pool3d(x, face)
        else:
            assert False
        return ret

    def channel_max_pool0d(self, x, face):
        assert self.ndim(x) == 2
        in_channels = self.shape(x)[1]
        assert 0 <= face <= in_channels
        assert in_channels % face == 0
        x = self.expand_dims(x, 1)
        pool_face = face
        pool_stride = pool_face
        pool_padding = 0
        x = F.max_pool1d(x, pool_face, pool_stride, pool_padding)
        return self.squeeze(x, 1)

    def channel_max_pool1d(self, x, face):
        assert self.ndim(x) == 3
        in_channels = self.shape(x)[1]
        assert 0 <= face <= in_channels
        assert in_channels % face == 0
        x = self.expand_dims(x, 1)
        pool_face = face, 1
        pool_stride = pool_face
        pool_padding = 0
        x = F.max_pool2d(x, pool_face, pool_stride, pool_padding)
        return self.squeeze(x, 1)

    def channel_max_pool2d(self, x, face):
        assert self.ndim(x) == 4
        in_channels = self.shape(x)[1]
        assert 0 <= face <= in_channels
        assert in_channels % face == 0
        x = self.expand_dims(x, 1)
        pool_face = face, 1, 1
        pool_stride = pool_face
        pool_padding = 0
        x = F.max_pool3d(x, pool_face, pool_stride, pool_padding)
        return self.squeeze(x, 1)

    def channel_max_pool3d(self, x, face):
        assert self.ndim(x) == 4
        in_channels = self.shape(x)[1]
        assert 0 <= face <= in_channels
        assert in_channels % face == 0
        pool_face = face, 1, 1
        pool_stride = pool_face
        pool_padding = 0
        xx = self.split(x, in_channels // face)
        for i, x in enumerate(xx):
            x = self.expand_dims(x, 1)
            x = F.max_pool3d(x, pool_face, pool_stride, pool_padding)
            x = self.squeeze(x, 1)
            xx[i] = x
        return self.concat(xx, axis=1)


class PyTorchChannelPoolAPI(BaseChannelPoolAPI, PyTorchChannelAvgPoolAPI,
                            PyTorchChannelMaxPoolAPI):
    pass
