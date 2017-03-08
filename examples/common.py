import sys
from pdt.common import *
from pdt.train_tf import *
from wacacore.util.io import handle_args

def boolify(x):
    if x in ['0', 0, False, 'False', 'false']:
        return False
    elif x in ['1', 1, True, 'True', 'true']:
        return True
    else:
        assert False, "couldn't convert to bool"

def handle_options(adt, argv):
    parser = PassThroughOptionParser()
    parser.add_option('-t', '--template', dest='template', nargs=1, type='string')
    (poptions, args) = parser.parse_args(argv)
    options = {}
    if poptions.template is None:
        options['template'] = 'res_net'
    else:
        options['template'] = poptions.template
    template_kwargs = template_module[options['template']].kwargs()
    options.update(template_kwargs)
    options['train'] = (boolify, 1)
    options['nitems'] = (int, 3)
    options['width'] = (int, 28)
    options['height'] = (int, 28)
    options['num_iterations'] = (int, 100)
    options['save_every'] = (int, 100)
    options['batch_size'] = (int, 512)
    options['compress'] = (boolify, 0)
    options['compile_fns'] = (boolify, 1)
    options['adt'] = (str, adt)
    options = handle_args(argv, options)
    options['template'] = template_module[options['template']].template
    return options
