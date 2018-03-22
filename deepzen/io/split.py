import numpy as np


class Split(object):
    def num_samples(self):
        raise NotImplementedError

    def sample(self, index):
        raise NotImplementedError

    def shapes(self, batch_size=None):
        xx, yy = self.sample(0)
        if batch_size is None:
            x_shapes = [x.shape for x in xx]
            y_shapes = [y.shape for y in yy]
        else:
            x_shapes = [(batch_size,) + x.shape for x in xx]
            y_shapes = [(batch_size,) + y.shape for y in yy]
        return x_shapes, y_shapes

    def dtypes(self):
        xx, yy = self.sample(0)
        x_dtypes = tuple(map(lambda x: x.dtype.name, xx))
        y_dtypes = tuple(map(lambda y: y.dtype.name, yy))
        return x_dtypes, y_dtypes

    def num_batches(self, batch_size):
        return self.num_samples() // batch_size

    def each_batch(self, batch_size):
        # Randomly assign samples to batches.  The remainders get dropped.
        num_samples = self.num_samples()
        sample_indices = np.arange(num_samples)
        np.random.shuffle(sample_indices)
        num_batches = num_samples // batch_size

        # Allocate the ndarrays that we will load the samples into.
        x_shapes, y_shapes = self.shapes(batch_size)
        x_dtypes, y_dtypes = self.dtypes()
        xxx = []
        for shape, dtype in zip(x_shapes, x_dtypes):
            xx = np.zeros(shape, dtype)
            xxx.append(xx)
        yyy = []
        for shape, dtype in zip(y_shapes, y_dtypes):
            yy = np.zeros(shape, dtype)
            yyy.append(yy)

        # Iterate the batches, loading their samples and yielding them.
        for batch in range(num_batches):
            a = batch * batch_size
            z = (batch + 1) * batch_size
            for i, sample_index in enumerate(sample_indices[a:z]):
                sample_xx, sample_yy = self.sample(sample_index)
                for j, x in enumerate(sample_xx):
                    xxx[j][i] = x
                for j, y in enumerate(sample_yy):
                    yyy[j][i] = y
            yield xxx.copy(), yyy.copy()
