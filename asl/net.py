import asl
from asl.archs.mlp import MLPNet
import torch.nn as nn

def type_check(xs, types):
  "Check the types and sizes are consistent"
  assert len(xs) == len(types)
  for i, x in enumerate(xs):
    same_size = x.size()[1:] == types[i].typesize
    assert same_size, "{} != {}".format(x.size()[1:], types[i].typesize)
  return xs


class Net(nn.Module):
  "A neural network"
  def __init__(self, name,
               module=None,
               arch=MLPNet,       # Architecture to use
               arch_opt=None,
               sample=False,      # If true will sample hyperparams
               sample_args=None): # arguments used to sample hyper params
    super(Net, self).__init__()
    if module is None:
      arch_opt = {} if arch_opt is None else arch_opt
      if sample:
        sample_args = {} if sample_args is None else sample_args
        arch_opt = arch.sample_hyper(self.in_sizes(), self.out_sizes(), **sample_args)
      self.module = arch(self.in_sizes(), self.out_sizes(), **arch_opt)
    else:
      self.module = module
    self.add_module(name, self.module)

  def forward(self, *xs):
    args = type_check(xs, self.in_types)
    argvals = asl.expand_consts([arg.value for arg in args])
    res = self.module.forward(*argvals)
    type_check(res, self.out_types)
    return [self.out_types[i](res[i]) for i in range(len(res))]
