"Templates (modules parameterized by shape)"
from asl.modules.modules import expand_consts, ModuleDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from asl.util.misc import mul_product
from asl.templates.packing import unstack_channel, cat_channels

class VarConvNet(nn.Module):
  "ConvNet which takes variable inputs and variable outputs"

  def __init__(self, in_sizes, out_sizes,
               channel_dim=1,
               batch_norm=False,
               h_channels=16,
               nhlayers=24,
               combine_inputs=cat_channels,
               activation=F.elu):
    super(VarConvNet, self).__init__()
    # Assumes batch not in size and all in/out same size except channel
    self.in_sizes = in_sizes
    self.out_sizes = out_sizes
    self.channel_dim = channel_dim
    self.activation = activation
    self.combine_inputs = combine_inputs
    ch_dim_wo_batch = channel_dim - 1
    in_channels = sum([size[ch_dim_wo_batch] for size in in_sizes])
    out_channels = sum([size[ch_dim_wo_batch] for size in out_sizes])

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

  def sample_hyper(in_sizes, out_sizes, pbatch_norm=0.5, max_layers=5):
    "Hyper Parameter Sampler"
    batch_norm = np.random.rand() > pbatch_norm
    nlayers = np.random.randint(1, max_layers)
    h_channels = int(np.random.choice([12, 16, 24]))
    act = np.random.choice([F.relu, F.elu])
    return {'batch_norm': batch_norm,
            'h_channels': h_channels,
            'nhlayers': nlayers,
            'activation': act}

  def forward(self, *xs):
    assert len(xs) == len(self.in_sizes), "Wrong # inputs"
    exp_xs = expand_consts(xs) # TODO: MOVE THIS TO FUNC,
    # Combine inputs
    x = self.combine_inputs(exp_xs)
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
    return unstack_channel(x, self.out_sizes)
