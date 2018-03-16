from ..base.form import Form
from ..base.layer import Layer
from ..base.spec import Spec


class DataLayer(Layer):
    def __init__(self, form):
        self.form = form

    def forward(self, x):
        self.form.check(x)
        return x


class DataSpec(Spec):
    def __init__(self, shape, dtype):
        self.form = Form(shape, dtype)

    def build(self, form=None):
        if form is None:
            form = self.form
        else:
            assert self.form.equals(form)
        return DataLayer(form), form
