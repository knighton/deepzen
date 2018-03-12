import json
import numpy as np

from sunyata import api as Z
from sunyata.crit.loss import CategoricalCrossEntropy
from sunyata.crit.metric import CategoricalAccuracy
from sunyata.dataset.mnist import load_mnist
from sunyata.layer import *  # noqa
from sunyata.model import Model
from sunyata.optim import *  # noqa


dataset = load_mnist()
x_sample = dataset[0][0][0]
y_sample = dataset[0][1][0]
y_dim, = y_sample.shape
spec = SequenceSpec([
    DataSpec(x_sample.shape, x_sample.dtype),
    FlattenSpec(),
    DenseSpec(256),
    ReLUSpec(),
    DenseSpec(64),
    ReLUSpec(),
    DenseSpec(y_dim),
    SoftmaxSpec(),
])
model = Model(spec)
model.fit_clf(dataset, callback='progress_bar')
