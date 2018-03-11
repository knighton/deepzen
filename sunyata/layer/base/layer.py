class Layer(object):
    def params(self):
        return []

    def forward(self, x):
        raise NotImplementedError
