import pdt.config

if pdt.config.backend == 'tensorflow':
    from pdt.backend.tensorflow.distances import *
elif pdt.config.backend == 'theano':
    from pdt.backend.theano.distances import *
