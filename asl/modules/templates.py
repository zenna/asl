import torch
import torch.nn as nn
from asl.modules.modules import expand_consts
import torch.nn.functional as F

# FIXME: Slice dim and channel dim should be same, confusing because off by 1
# adjustmenets due to batching, revise!
def unstack_channel(t, sizes, channel_dim=0, slice_dim=1):
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
      outputs.append(t.narrow(slice_dim, c0, c))
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

from functools import reduce # Valid in Python 2.6+, required in Python 3
import operator
def mul_product(xs):
  return reduce(operator.mul, xs, 1)

class MLPNet(nn.Module):
  "ConvNet which takes variable inputs and variable outputs"

  def __init__(self, in_sizes, out_sizes, channel_dim=1):
    super(MLPNet, self).__init__()
    # Assumes batch not in size and all in/out same size except channel
    self.flat_in_size = [mul_product(size) for size in in_sizes]
    self.flat_out_size = [mul_product(size) for size in out_sizes]

    self.nin = sum(self.flat_in_size)
    self.nout = sum(self.flat_out_size)
    self.in_sizes = in_sizes
    self.out_sizes = out_sizes
    self.m = nn.Linear(self.nin, self.nout)

  def forward(self, *xs):
    assert len(xs) == len(self.in_sizes), "Wrong # inputs"
    exp_xs = expand_consts(xs) # TODO: Make optional
    exp_xs = [x.contiguous().view(x.size(0), -1) for x in exp_xs]
    x = torch.cat(exp_xs, dim=1)
    y = self.m(x)
    outxs = unstack_channel(y, self.out_sizes)
    res = [x.contiguous().view(x.size(0), *self.out_sizes[i]) for (i, x) in enumerate(outxs)]
    return res
