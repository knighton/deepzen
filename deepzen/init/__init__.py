import sys

from .base.registry import get
from .fixed import *  # noqa
from .random import *  # noqa


unpack_initializer = get
