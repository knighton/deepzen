from .base.callback import Callback
from .progress_bar import ProgressBar


def unpack_callback(arg):
    if isinstance(arg, Callback):
        return arg
    klass = {
        'progress_bar': ProgressBar,
    }[arg]
    return klass()
    

def unpack_callbacks(arg):
    if arg is None:
       return []
    if not isinstance(arg, (list, tuple)):
        arg = [arg]
    callbacks = []
    for item in arg:
        callbacks.append(unpack_callback(item))
    return callbacks
