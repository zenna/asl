import sys
from pdt.train_tf import *
from wacacore.util.io import handle_args
from tensortemplates.module import template_module, nl_module
import os

def boolify(x):
    "Convert `x` into a Boolean"
    if x in ['0', 0, False, 'False', 'false']:
        return False
    elif x in ['1', 1, True, 'True', 'true']:
        return True
    else:
        assert False, "couldn't convert %s  to bool" % x

def default_options():
    "Get default options for pdt training"
    options = {}
    options['num_iterations'] = (int, 100)
    options['save_every'] = (int, 100)
    options['batch_size'] = (int, 512)
    options['dirname'] = (str, "dirname")
    options['datadir'] = (str, os.path.join(os.environ['DATADIR'], "pdt"))
    return options

def handle_options(adt, argv):
    parser = PassThroughOptionParser()
    parser.add_option('-t', '--template', dest='template', nargs=1, type='string')
    (poptions, args) = parser.parse_args(argv)
    options = {}
    if poptions.template is None:
        options['template'] = 'res_net'
    else:
        options['template'] = poptions.template
    template_options = template_module[options['template']].kwargs()
    options.update(template_options)
    options.update(default_options())
    options['adt'] = (str, adt)
    options = handle_args(argv, options)
    options['template'] = template_module[options['template']].template
    options['nl'] = nl_module[options['nl']]
    return options
