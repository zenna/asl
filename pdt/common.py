# Functions common for examples
from pdt.io import *
import tensortemplates as tt
from tensortemplates.res_net import *
# from pdt.templates.conv_net import *
# from pdt.templates.res_net import *
# from pdt.templates.warp_conv_net import *


def gen_sfx_key(keys, options):
    sfx_dict = {}
    for key in keys:
        sfx_dict[key] = options[key]
    sfx = stringy_dict(sfx_dict)
    print("sfx:", sfx)
    return sfx


def parse_template(template):
    if template == 'res_net':
        return tt.res_net.template
    elif template == 'conv_net':
        return tt.conv_res_net.template
    else:
        print("Invalid Template ", template)
        raise ValueError


default_template_map = {'res_net': tt.res_net.kwargs}


def default_template_kwargs(template):
    """Return the arguments required for a particular template"""
    return default_template_map[template]()
