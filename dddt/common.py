# Functions common for examples
from dddt.io import *
import tensortemplates as tt
from tensortemplates.res_net import *
from tensortemplates.conv_res_net import *
# from dddt.templates.res_net import *
# from dddt.templates.warp_conv_net import *

def gen_sfx_key(keys, options):
    sfx_dict = {}
    for key in keys:
        sfx_dict[key] = options[key]
    sfx = stringy_dict(sfx_dict)
    print("sfx:", sfx)
    return sfx


# def parse_template(template):
#     if template == 'res_net':
#         return tt.res_net.res_net
#     elif template == 'conv_res_net':
#         return conv_res_net
#     elif template == 'warp_conv_net':
#         return warp_conv_net
#     else:
#         print("Invalid Template ", template)
#         raise ValueError

template_module = {'res_net': tt.res_net, 'conv_res_net': tt.conv_res_net}
