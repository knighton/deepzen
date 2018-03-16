class BaseReLUAPI(object):
    def relu(self, x):
        return self.clip(x, min=0)
