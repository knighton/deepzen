import os
import sys


name = os.environ.get('SUNYATA_BACKEND', 'pytorch')
print('Sunyata backend: %s.' % name)

if name == 'mxnet':
    from .mxnet import MXNetAPI as BackendAPI
elif name == 'pytorch':
    from .pytorch import PyTorchAPI as BackendAPI
else:
    assert False, \
        'Unsupported backend (selected: %s) (possible values: %s).' % \
            (name, ('mxnet', 'pytorch'))

api = BackendAPI()

module = sys.modules[__name__]
for method_name in filter(lambda s: not s.startswith('_'), dir(api)):
    setattr(module, method_name, getattr(api, method_name))
