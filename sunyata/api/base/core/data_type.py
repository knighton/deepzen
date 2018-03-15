from contextlib import contextmanager


class BaseDataTypeAPI(object):
    def __init__(self, dtypes, home_floatx):
        self._all_dtypes = set("""
             bool
                   int8   int16   int32   int64
                  uint8  uint16  uint32  uint64
                        float16 float32 float64
        """.split())
        is_floatx = lambda s: s.startswith('float')
        self._all_floatxs = set(filter(is_floatx, self._all_dtypes))

        assert isinstance(dtypes, set)
        self._floatxs = set()
        for dtype in dtypes:
            assert dtype in self._all_dtypes
            if is_floatx(dtype):
                self._floatxs.add(dtype)
        self._dtypes = dtypes

        assert home_floatx in self._floatxs
        self._floatx_scopes = [home_floatx]

    def dtypes(self):
        return self._dtypes

    def get_dtype(self, x=None):
        if x is None:
            x = self._floatx_scopes[-1]
        else:
            assert x in self._dtypes
        return x

    def dtype(self, x=None):
        raise NotImplementedError

    def floatxs(self):
        return self._floatxs

    def get_floatx(self, x=None):
        if x is None:
            x = self._floatx_scopes[-1]
        else:
            assert x in self._floatxs
        return x

    def floatx(self, x=None):
        raise NotImplementedError

    def set_home_floatx(self, x):
        self._floatx_scopes[0] = self.floatx(x)

    @contextmanager
    def floatx_scope(self, x):
        floatx = self.floatx(x)
        self._floatx_scopes.append(floatx)
        yield
        self._floatx_scopes.pop()
