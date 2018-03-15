class BaseReshapeAPI(object):
    def reshape(self, x, shape):
        raise NotImplementedError

    def flatten(self, x):
        return self.reshape(x, -1)

    def expand_dims(self, x, axis):
        raise NotImplementedError

    def squeeze(self, x, axis=None):
        raise NotImplementedError

    def repeat_axis(self, x, axis, repeat):
        splits = self.split(x, axis)
        xx = []
        for split in splits:
            for i in range(repeat):
                xx.append(split)
        return self.concat(xx, axis)

    def repeat(self, x, repeats):
        assert len(x.shape) == len(repeats)
        for i, repeat in enumerate(repeats):
            if repeat == 1:
                continue
            x = self.repeat_axis(x, i, repeat)
        return x

    def tile_axis(self, x, axis, repeat):
        splits = self.split(x, axis)
        xx = []
        for i in range(repeat):
            for split in splits:
                xx.append(split)
        return self.concat(xx, axis)

    def tile(self, x, repeats):
        assert len(x.shape) == len(repeats)
        for i, repeat in enumerate(repeats):
            if repeat == 1:
                continue
            x = self.tile_axis(x, i, repeat)
        return x

    def transpose(self, x, axes):
        raise NotImplementedError

    def split(self, x, axis):
        raise NotImplementedError

    def concat(self, xx, axis):
        raise NotImplementedError

    def stack(self, xx, axis=0):
        raise NotImplementedError
