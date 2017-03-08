"""Functions common for examples"""
from wacacore.util.misc import stringy_dict
import tensortemplates.res_net as res_net
import tensortemplates.conv_res_net as conv_res_net
import tensortemplates as tt


def gen_sfx_key(keys, options):
    sfx_dict = {}
    for key in keys:
        sfx_dict[key] = options[key]
    sfx = stringy_dict(sfx_dict)
    print("sfx:", sfx)
    return sfx


template_module = {'res_net': res_net, 'conv_res_net': conv_res_net}
