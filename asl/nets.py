import torch
import torch.nn as nn
import torch.nn.functional as F
from asl.type import expand_consts


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
    xs = expand_consts(xs) # TODO: Make optional
    x = torch.cat(xs, dim=self.channel_dim)
    # Combine inputs
    # Middle Layers
    x = F.elu(self.conv1(x))
    x = F.elu(self.conv2(x))
    # Uncombine
    return unstack_channel(x, self.out_sizes)
