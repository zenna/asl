from asl.templates.mlp import MLPNet
import torch.nn as nn

def type_check(xs, types):
  assert len(xs) == len(types)
  for i, x in enumerate(xs):
    same_size = xs[i].size()[1:] == types[i].size
    assert same_size
  return xs


class Net(nn.Module):
  def __init__(self, name, module=None, template=MLPNet, template_opt=None):
    super(Net, self).__init__()
    if module is None:
      template_opt = {} if template_opt is None else template_opt
      self.module = template(self.in_sizes(), self.out_sizes(), **template_opt)
    else:
      self.module = module
    self.add_module(name, self.module)

  def forward(self, *xs):
    args = type_check(xs, self.in_types)
    res = self.module.forward(*args)
    return type_check(res, self.out_types)
