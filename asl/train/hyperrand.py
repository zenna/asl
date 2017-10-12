"""Hyper Parameter Search"""
import time
from wacacore.util.misc import extract, dict_prod
import numpy as np


def rand_product(run_me, options, var_option_keys, nsamples, prefix='',
                 nrepeats=1):
    """Train parametric inverse and vanilla neural network with different
    amounts of data and see the test_error
    Args:
        run_me: function to call, should execute test and save stuff
        options: Options to be passed into run_me
        var_option_keys: Set of keys, where options['keys'] is a sequence
            and we will vary over cartesian product of all the keys
        nsamples: sample nsamples hyperparameter values
        prefix: string prefix for this job
        nrepeats: for every set of hyper parameters, repeat nrepeats times
    """
    _options = {}
    _options.update(options)
    var_options = extract(var_option_keys, options)

    var_options_prod = list(dict_prod(var_options))
    for i in range(nsamples):
        if len(var_options_prod) == 0:
            break
        prod = np.random.choice(var_options_prod, replace=False)
        for j in range(nrepeats):
            the_time = time.time()
            dirname = "%s_%s_%s_%s" % (prefix, str(the_time), i, j)
            _options['dirname'] = dirname
            _options.update(prod)
            run_me(_options)

def test_everything(run_me, options, var_option_keys, prefix='', nrepeats=1):
    """Train parametric inverse and vanilla neural network with different
    amounts of data and see the test_error
    Args:
        run_me: function to call, should execute test and save stuff
        Options: Options to be passed into run_me
        var_option_keys: Set of keys, where options['keys'] is a sequence
            and we will vary over cartesian product of all the keys

    """
    _options = {}
    _options.update(options)
    var_options = extract(var_option_keys, options)

    for i in range(nrepeats):
        var_options_prod = dict_prod(var_options)
        the_time = time.time()
        for j, prod in enumerate(var_options_prod):
            dirname = "%s_%s_%s_%s" % (prefix, str(the_time), i, j)
            _options['dirname'] = dirname
            _options.update(prod)
            run_me(_options)
