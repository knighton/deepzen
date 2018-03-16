class Layer(object):
    def params(self):
        return []

    def forward(self, x, is_training):
        raise NotImplementedError
