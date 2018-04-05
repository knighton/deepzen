from copy import deepcopy

from .pseudo_node import PseudoNode
from .spec import Spec


class Keyword(PseudoNode):
    def __init__(self, spec_class, default_kwargs=None):
        assert issubclass(spec_class, Spec)
        if default_kwargs is None:
            default_kwargs = {}
        else:
            assert isinstance(default_kwargs, dict)
        PseudoNode.__init__(self)
        self._spec_class = spec_class
        self._default_kwargs = default_kwargs

    def copy(self):
        spec_class = self._spec_class
        default_kwargs = deepcopy(self._default_kwargs)
        return Keyword(spec_class, default_kwargs)

    def desugar(self):
        return self.__call__()

    def __call__(self, *args, **override_kwargs):
        from .atom import Atom
        kwargs = deepcopy(self._default_kwargs)
        kwargs.update(deepcopy(override_kwargs))
        spec = self._spec_class(*args, **kwargs)
        return Atom(spec)


def keywordize(spec_class, xsnd=None, default_kwargs=None):
    if not isinstance(xsnd, list):
        xsnds = [xsnd]
    else:
        xsnds = xsnd
    if default_kwargs is None:
        default_kwargs = {}
    rr = []
    for xsnd in xsnds:
        kwargs = deepcopy(default_kwargs)
        if xsnd is not None:
            assert 'xsnd' not in kwargs
            kwargs['xsnd'] = xsnd
        r = Keyword(spec_class, default_kwargs)
        rr.append(r)
    if len(rr) == 1:
        return rr[0]
    else:
        assert rr
        return rr
