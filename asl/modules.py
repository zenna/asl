import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from asl.util import is_tensor_var
from asl.type import Constant


def unstack_channel(t, sizes, channel_dim=0):
  assert len(sizes) > 0
  channels = [size[channel_dim] for size in sizes]
  if len(sizes) == 1:
    # print("Only one output skipping unstack")
    return (t,)
  else:
    outputs = []
    c0 = 0
    for c in channels:
      # print("Split ", c0, ":", c0 + c)
      outputs.append(t[:, c0:c0 + c, :, :])
      c0 = c

  return tuple(outputs)

class VarConvNet(nn.Module):
  "ConvNet which takes variable inputs and variable outputs"

  def __init__(self, in_sizes, out_sizes, channel_dim=1):
    super(VarConvNet, self).__init__()
    # Assumes batch not in size and all in/out same size except channel
    self.in_sizes = in_sizes
    self.out_sizes = out_sizes
    self.channel_dim = channel_dim
    ch_dim_wo_batch = channel_dim - 1
    in_channels = sum([size[ch_dim_wo_batch] for size in in_sizes])
    out_channels = sum([size[ch_dim_wo_batch] for size in out_sizes])
    mid_channel = 16
    self.conv1 = nn.Conv2d(in_channels, mid_channel, 3, padding=1)
    self.convmid = nn.Conv2d(mid_channel, mid_channel, 3, padding=1)
    self.conv2 = nn.Conv2d(mid_channel, out_channels, 3, padding=1)

  def forward(self, *xs):
    assert len(xs) == len(self.in_sizes), "Wrong # inputs"
    exp_xs = expand_consts(xs) # TODO: Make optional
    x = torch.cat(exp_xs, dim=self.channel_dim)
    # Combine inputs
    # Middle Layers
    x = F.elu(self.conv1(x))
    x = F.elu(self.conv2(x))
    # Uncombine
    return unstack_channel(x, self.out_sizes)


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
