import dddt.config

if dddt.config.backend == 'tensorflow':
    from dddt.backend.tensorflow.distances import *
elif dddt.config.backend == 'theano':
    from dddt.backend.theano.distances import *
