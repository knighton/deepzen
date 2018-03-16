import numpy as np

from ... import api as Z
from ..base.form import Form
from ..base.layer import Layer
from ..base.spec import Spec


class FlattenLayer(Layer):
    def forward(self, x, is_training):
        return Z.batch_flatten(x)


class FlattenSpec(Spec):
    def build(self, form=None):
        out_shape = int(np.prod(form.shape)),
        return FlattenLayer(), Form(out_shape, form.dtype)
