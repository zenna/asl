import torch
from torch.autograd import Variable
from asl.util import is_tensor_var

class Type:
  """A type is a Type"""
  def __init__(self, name, size, dtype, observable=True):
    self.name = name
    self.size = size
    self.dtype = dtype
    self.observable = observable

  def size(self):
    return self.size


class FunctionType:
  "Function Type"

  def __init__(self, in_types, out_types):
    self.in_types = in_types
    self.out_types = out_types


class Function():
  "Typed Function"

  def __init__(self, in_types, out_types):
    super(Function, self).__init__()
    self.in_types = in_types
    self.out_types = out_types

  def type(self):
    "Function Type"
    return FunctionType(self.in_types, self.out_types)

  def in_sizes(self):
    return [type.size for type in self.type().in_types]

  def out_sizes(self):
    return [type.size for type in self.type().out_types]

  def n_inputs(self):
    return len(self.type().in_types)

  def n_outputs(self):
    return len(self.type().out_types)


class Constant:
  "Typed Constant"

  def __init__(self, type):
    self.type = type
    self._size = (1, ) + type.size
    self.value = Variable(torch.rand(*self._size).cuda(), requires_grad=True)

  def size(self):
    return self._size

  def parameters(self):
    return [self.value]

  def expand_to_batch(self, batch_size):
    expanded_size = (batch_size, ) + self.type.size
    return self.value.expand(expanded_size)

  def __call__(self):
    return self.value.expand(*self.expanded_size)


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
