class BaseTypeAPI(object):
    def scalar(self, x):
        x = self.numpy(x)
        assert x.size == 1
        return x.flatten()[0]

    def list(self, x):
        x = self.numpy(x)
        return x.tolist()

    def numpy(self, x):
        raise NotImplementedError

    def tensor(self, x, dtype=None, device=None):
        raise NotImplementedError

    def constant(self, x, dtype=None, device=None):
        raise NotImplementedError

    def variable(self, x, dtype=None, device=None):
        raise NotImplementedError
