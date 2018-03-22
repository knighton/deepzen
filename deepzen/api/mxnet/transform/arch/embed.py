import mxnet as mx

from ....base.transform.arch.embed import BaseEmbedAPI


class MXNetEmbedAPI(BaseEmbedAPI):
    def embed(self, x, table):
        vocab_size, channels = table.shape
        channels_last = mx.nd.Embedding(
            data=x, weight=table, input_dim=vocab_size, output_dim=channels,
            dtype=table.dtype)
        ndim = channels_last.ndim
        axes = (0, ndim - 1) + tuple(range(1, ndim - 1))
        return mx.nd.transpose(channels_last, axes)
