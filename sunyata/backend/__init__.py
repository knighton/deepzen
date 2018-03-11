import os
import sys


name = os.environ.get('SUNYATA_BACKEND', 'pytorch')
print('Sunyata backend: %s.' % name)

if name == 'mxnet':
    from .mxnet import MXNetBackend as Backend
elif name == 'pytorch':
    from .pytorch import PyTorchBackend as Backend
else:
    assert False, \
        'Unsupported backend (selected: %s) (possible values: %s).' % \
            (name, ('mxnet', 'pytorch'))

backend = Backend()

module = sys.modules[__name__]
for method_name in filter(lambda s: not s.startswith('_'), dir(backend)):
    setattr(module, method_name, getattr(backend, method_name))
