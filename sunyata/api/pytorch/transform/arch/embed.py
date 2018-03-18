from ....base.transform.arch.embed import BaseEmbedAPI


class PyTorchEmbedAPI(BaseEmbedAPI):
    def embed(self, x, table):
        channels_last = table.index_select(0, x.view(-1))
        channels_last = channels_last.view(x.size() + (-1,))
        ndim = channels_last.dim()
        axes = (0, ndim - 1) + tuple(range(1, ndim - 1))
        return channels_last.permute(*axes).contiguous()
