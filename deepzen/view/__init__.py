from .base.view import View
from .progress_bar import ProgressBar
from .row_per_epoch import RowPerEpoch
from .server import Server


def unpack_view(arg):
    if isinstance(arg, View):
        return arg
    klass = {
        'row_per_epoch': RowPerEpoch,
        'progress_bar': ProgressBar,
        'server': Server,
    }[arg]
    return klass()


def unpack_views(arg):
    if arg is None:
        return []
    if isinstance(arg, str):
        arg = arg.split(',')
    if not isinstance(arg, (list, tuple)):
        arg = [arg]
    views = []
    for item in arg:
        views.append(unpack_view(item))
    return views
