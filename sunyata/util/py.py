def require_kwargs(func):
    """
    Function decorator that requires all arguments to be passed as kwargs.

    in:
        function  func  The function to wrap.
    """

    def make_wrap(func):
        def wrap(*args, **kwargs):
            assert not args
            return func(**kwargs)
        wrap.__name__ = func.__name__
        return wrap
    return make_wrap(func)
