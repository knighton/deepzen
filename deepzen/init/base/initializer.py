from ...util.py import class_path


class Initializer(object):
    @classmethod
    def from_json(cls, x):
        return cls(**x)

    def __call__(self, shape, dtype, meaning=None):
        raise NotImplementedError

    def params_to_json(self):
        return self.__dict__

    def to_json(self):
        return {
            'name': class_path(self),
            'params': self.params_to_json(),
        }
