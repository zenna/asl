import torch
import torch.nn as nn
from torch.autograd import Variable
from asl.util.misc import is_tensor_var
from asl.type import Constant


class ModuleDict(nn.Module):

  def __init__(self, module_dict):
    super(ModuleDict, self).__init__()
    self.module_dict = module_dict
    for name, module in module_dict.items():
      self.add_module(name, module)

  def __getitem__(self, name):
    return self.module_dict[name]

  def __setitem__(self, name, value):
    self.add_module(name, value)
    self.module_dict[name] = value

  def keys(self):
    return self.module_dict.keys()

  def values(self):
    return self.module_dict.values()

  def items(self):
    return self.module_dict.items()


class ConstantNet(Constant, nn.Module):
  "Typed Constant"

  def __init__(self, type, requires_grad=True):
    super(ConstantNet, self).__init__(type)
    nn.Module.__init__(self)
    self._size = (1, ) + type.size
    self.value = nn.Parameter(torch.rand(*self._size),
                              requires_grad=requires_grad)

  def size(self):
    return self._size

  def expand_to_batch(self, batch_size):
    expanded_size = (batch_size, ) + self.type.size
    return self.value.expand(expanded_size)

  def forward(self):
    return self.value


def anybatchsize(args, batch_dim=0):
  for arg in args:
    if is_tensor_var(arg):
      return arg.size()[batch_dim]
  raise ValueError


def expand_consts(args):
  batch_size = anybatchsize(args)
  res = []

  for arg in args:
    if isinstance(arg, Constant):
      res.append(arg.expand_to_batch(batch_size))
    else:
      res.append(arg)
  return tuple(res)
