                    DeepZen
                    -------

    Directories
    -----------

api/ -- backend functional API
    base/ -- base backend mixins extended by the mxnet/pytorch versions
        core/ -- core operations
        layer/ -- functional equivalents of nodes
        meter/ -- implement losses and accuracies
    mxnet/
    pytorch/
app/ -- applications, eg VGG16
data/ -- dataset classes, used by models
init/ -- weight initializers (like Keras)
meter/ -- losses and accuracies
model/ -- model classes
node/ -- static computational graph nodes, that wrap backend API
    activ/ -- activation functions
    arch/ -- embedding, input, network, sequence
    base/ -- base types
    dot/ -- dense and conv
    merge/ -- multi-input, single-output
    norm/ -- batch norm, dropout, etc.
    shape/ -- nodes that reshape
optim/ -- optimizers (like PyTorch)
spy/ -- callback-based training monitors
task/ -- example datasets
transform/ -- raw data <-> tensors
util/ -- misc utils
