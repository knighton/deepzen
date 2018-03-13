from .base.callback import Callback
from .row_per_epoch import RowPerEpoch
from .progress_bar import ProgressBar


def unpack_callback(arg):
    if isinstance(arg, Callback):
        return arg
    klass = {
        'row_per_epoch': RowPerEpoch,
        'progress_bar': ProgressBar,
    }[arg]
    return klass()
    

def unpack_callbacks(arg):
    if arg is None:
        return []
    if isinstance(arg, str):
        arg = arg.split(',')
    if not isinstance(arg, (list, tuple)):
        arg = [arg]
    callbacks = []
    for item in arg:
        callbacks.append(unpack_callback(item))
    return callbacks
