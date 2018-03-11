def require_kwargs(func):
    """
    Decorator that requires all arguments be passed as kwargs.

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


def require_kwargs_after(num_args):
    """
    Decorator that requires all but the first "num_args" args be kwargs.

    in:
        int  num_args  All but the first num_args args must be kwargs.
    """

    def make_wrap(func):
        def wrap(*args, **kwargs):
            assert len(args) <= num_args
            return func(*args, **kwargs)
        wrap.__name__ = func.__name__
        return wrap
    return make_wrap
