from copy import deepcopy

from .pseudo_node import PseudoNode
from .spec import Spec


class Keyword(PseudoNode):
    """
    Instances of this class act like subclasses of Atom.

    They are instances, not classes, for the * and > Sequence construction
    syntactic sugar to work when they are plugged in directly, without
    parentheses.  In that case they still have to be objects.  So we have global
    Atom factory instances named like classes, which return Atoms when called.

    The vast majority of node names used to construct neural networks are
    Keywords.  Most of them do not require any explicit arguments due to shape
    inference.  There we can just pass the "class name".  It's much cleaner.

    Instances of this class create orphan Atoms when called, according to the
    Spec class and any Spec default arguments given during initialization.
    Keywords that were not called in the definition are called implicitly during
    Sequence __init__.  The Layer, Spec, and Keyword(s) constitute the three
    parts of the definition of a typical "layer" concept.

    They are created using the "keywordize" function below.
    """

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
        """
        Inherited from PseudoNode.
        """
        spec_class = self._spec_class
        default_kwargs = deepcopy(self._default_kwargs)
        return Keyword(spec_class, default_kwargs)

    def desugar(self):
        """
        Inherited from PseudoNode.
        """
        return self.__call__()

    def __call__(self, *args, **override_kwargs):
        """
        Instantiate an Atom according to the Spec params (fake Atom __init__).
        """
        from .atom import Atom
        kwargs = deepcopy(self._default_kwargs)
        kwargs.update(deepcopy(override_kwargs))
        spec = self._spec_class(*args, **kwargs)
        return Atom(spec)


def keywordize(spec_class, xsnd=None, default_kwargs=None):
    """
    Create the official node keyword(s) for the given Spec/Layer.

    Takes a Spec, optionally a list of xsnds (spatial ndim restrictions on the
    input tensor(s)), and further optionally any default arguments to be given
    to Spec init.  Creates a Keyword per xsnd.  Returns them in a list if there
    are multiple objects, else just the one.

        [in]

    * spec_class      Spec class         The spec class that it will
                                         instantiate and wrap inside a
                                         LayerNode when called.

    * xsnd            {None, int, list}  Spatial ndim restriction(s) on input.

    * default_kwargs  dict               Default args passed to Spec init.

        [out]

    * keywords  {Keyword, list<Keyword>}  The Keyword(s) created.
    """
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
