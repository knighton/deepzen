from .base.hook import Hook
from .progress_bar import ProgressBar
from .row_per_epoch import RowPerEpoch
from .server import Server


def unpack_hook(arg):
    if isinstance(arg, Hook):
        return arg
    klass = {
        'row_per_epoch': RowPerEpoch,
        'progress_bar': ProgressBar,
        'server': Server,
    }[arg]
    return klass()


def unpack_hooks(arg):
    if arg is None:
        return []
    if isinstance(arg, str):
        arg = arg.split(',')
    if not isinstance(arg, (list, tuple)):
        arg = [arg]
    hooks = []
    for item in arg:
        hooks.append(unpack_hook(item))
    return hooks
