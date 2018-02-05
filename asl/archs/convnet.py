"Templates (modules parameterized by shape)"
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from asl.archs.packing import split_channel, cat_channels, slither
from asl.modules.modules import expand_consts


def channels(sizes):
  total = 0
  for size in sizes:
    if len(size) == 1:
      total += 1
    elif len(size) == 3:
      total += size[0]
    else:
      print("Only sizes of 2 and 4 supported")
      raise ValueError
  return total


class ConvNet(nn.Module):
  "ConvNet which takes variable inputs and variable outputs"

  def sample_hyper(in_sizes, out_sizes, pbatch_norm=0.5, max_layers=5):
    "Sample hyper parameters"
    batch_norm = np.random.rand() > pbatch_norm
    nlayers = np.random.randint(1, max_layers)
    h_channels = random.choice([12, 16, 24])
    act = np.random.choice([F.relu, F.elu])
    return {'batch_norm': batch_norm,
            'h_channels': h_channels,
            'nhlayers': nlayers,
            'activation': act}

  def __init__(self,
               in_sizes,
               out_sizes,
               channel_dim=1,
               batch_norm=False,
               h_channels=8,
               nhlayers=4,
               combine_inputs=cat_channels,
               activation=F.elu):
    super(ConvNet, self).__init__()
    # Assumes batch not in size and all in/out same size except channel
    self.in_sizes = in_sizes
    self.out_sizes = out_sizes
    self.channel_dim = channel_dim
    self.activation = activation
    self.combine_inputs = combine_inputs
    in_channels = channels(in_sizes)
    out_channels = channels(out_sizes)
    # import pdb; pdb.set_trace()
    h_channels=8
    nhlayers=4
    activation=F.elu

    batch_norm = False

    # Layers
    self.conv1 = nn.Conv2d(in_channels, h_channels, 3, padding=1)
    hlayers = [nn.Conv2d(h_channels, h_channels, 3, padding=1) for i in range(nhlayers)]
    self.hlayers = nn.ModuleList(hlayers)

    # Batch norm
    self.batch_norm = batch_norm
    if batch_norm:
      blayers = [nn.BatchNorm2d(h_channels, affine=False) for i in range(nhlayers)]
      self.blayers = nn.ModuleList(blayers)

    self.conv2 = nn.Conv2d(h_channels, out_channels, 3, padding=1)

  def forward(self, *xs):
    assert len(xs) == len(self.in_sizes), "Wrong # inputs"
    x = self.combine_inputs(xs)
    x = self.conv1(x)

    # h layers
    for (i, layer) in enumerate(self.hlayers):
      x = layer(x)
      if self.batch_norm:
        x = self.blayers[i](x)
      x = self.activation(x)

    x = self.conv2(x)
    x = self.activation(x)

    # Uncombine
    split_channels = split_channel(x, self.out_sizes)
    return tuple(map(slither, split_channels, self.out_sizes)) # FIXME, not all men
