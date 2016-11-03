import pdt.config

# Backend specific
if pdt.config.backend == 'tensorflow':
    from pdt.backend.tensorflow.io import *
elif pdt.config.backend == 'theano':
    import pdt.backend.theano.io
