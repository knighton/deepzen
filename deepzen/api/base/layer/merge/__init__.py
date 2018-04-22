class BaseMergeAPI(object):
    def merge_mean(self, xx):
        assert xx
        y = self.copy(xx[0])
        for x in xx[1:]:
            y += x
        return y / len(xx)

    def merge_concat(self, xx, sample_axis):
        batch_axis = 1 + sample_axis
        return self.concat(xx, batch_axis)

    def merge_difference(self, xx):
        assert 2 <= len(xx)
        y = self.copy(xx[0])
        for x in xx[1:]:
            y -= x
        return y

    def merge_maximum(self, xx):
        assert xx
        y = self.copy(xx[0])
        for x in xx[1:]:
            y = self.maximum(x, y)
        return y

    def merge_minimum(self, xx):
        assert xx
        y = self.copy(xx[0])
        for x in xx[1:]:
            y = self.minimum(x, y)
        return y

    def merge_product(self, xx):
        assert xx
        y = self.copy(xx[0])
        for x in xx[1:]:
            y *= x
        return y

    def merge_stack(self, xx, sample_axis):
        batch_axis = 1 + sample_axis
        return self.stack(xx, batch_axis)

    def merge_sum(self, xx):
        assert xx
        y = self.copy(xx[0])
        for x in xx[1:]:
            y += x
        return y
