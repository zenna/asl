# Functions common for examples
from pdt.io import *
import tensortemplates as tt
from tensortemplates.res_net import *
from tensortemplates.conv_res_net import *
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


template_module = {'res_net': tt.res_net, 'conv_res_net': tt.conv_res_net}
