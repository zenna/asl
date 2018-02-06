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

def constant_test(x):
  return nn.init.constant(x, val =0.1)

def normal_trunc(x):
  return nn.init.normal(x, std=0.1)


class ConvNet(nn.Module):
  "ConvNet which takes variable inputs and variable outputs"

  def sample_hyper(in_sizes, out_sizes, pbatch_norm=0.5, max_layers=8):
    "Sample hyper parameters"
    batch_norm = np.random.rand() > pbatch_norm
    learn_batch_norm = np.random.rand() > 0.5
    nlayers = np.random.randint(0, max_layers)
    h_channels = random.choice([4, 8, 12, 16, 24])
    act = random.choice([F.elu])
    last_act = random.choice([F.elu])
    ks = random.choice([3, 5])
    conv_init = random.choice([nn.init.xavier_uniform])
    return {'batch_norm': batch_norm,
            'h_channels': h_channels,
            'nhlayers': nlayers,
            'activation': act,
            'ks': ks,
            'last_activation': last_act,
            'learn_batch_norm': learn_batch_norm,
            'padding': (ks - 1)//2,
            'conv_init': conv_init}

  # def sample_hyper(in_sizes, out_sizes, pbatch_norm=0.5, max_layers=5):
  #   "Sample hyper parameters"
  #   batch_norm = np.random.rand() > pbatch_norm
  #   batch_norm = False
  #   learn_batch_norm = np.random.rand() > 0.5
  #   # nlayers = np.random.randint(0, max_layers)
  #   nlayers = 4
  #   h_channels = random.choice([8])
  #   act = random.choice([F.elu])
  #   last_act = random.choice([F.elu])
  #   ks = random.choice([3])
  #   conv_init = normal_trunc
  #   # bias_init = constant_test
  #   return {'batch_norm': batch_norm,
  #           'h_channels': h_channels,
  #           'nhlayers': nlayers,
  #           'activation': act,
  #           'ks': ks,
  #           'last_activation': last_act,
  #           'learn_batch_norm': learn_batch_norm,
  #           'padding': (ks - 1)//2,
  #           'conv_init': conv_init}

  def __init__(self,
               in_sizes,
               out_sizes,
               channel_dim=1,
               batch_norm=False,
               h_channels=8,
               nhlayers=4,
               ks=3,
               combine_inputs=cat_channels,
               activation=F.elu,
               last_activation=F.elu,
               learn_batch_norm=True,
               padding=2,
               conv_init=nn.init.xavier_uniform):
    super(ConvNet, self).__init__()
    # Assumes batch not in size and all in/out same size except channel
    self.in_sizes = in_sizes
    self.out_sizes = out_sizes
    self.channel_dim = channel_dim
    self.activation = activation
    self.last_activation = last_activation
    self.combine_inputs = combine_inputs
    self.batch_norm = batch_norm
    in_channels = channels(in_sizes)
    out_channels = channels(out_sizes)
    # Layers
    hlayers = [nn.Conv2d(h_channels, h_channels, ks, padding=padding) for i in range(nhlayers)]
    self.hlayers = nn.ModuleList(hlayers)
    self.conv1 = nn.Conv2d(in_channels, h_channels, ks, padding=padding)

    # Batch norm
    if batch_norm:
      firstblayer = nn.BatchNorm2d(h_channels, affine=False)
      lastblayer = nn.BatchNorm2d(out_channels, affine=False)
      hblayers = [nn.BatchNorm2d(h_channels, affine=False) for i in range(nhlayers)]
      self.firstblayer = firstblayer
      self.lastblayer = lastblayer
      self.hblayers = hblayers
      allblayers = [firstblayer] + hblayers + [lastblayer]
      self.allblayers = nn.ModuleList(allblayers)
      if not learn_batch_norm:
        self.allblayers.eval() # Turn to eval mode to not learn parameters

    self.conv2 = nn.Conv2d(h_channels, out_channels, ks, padding=padding)

    # Init
    # for convlayer in [self.conv1, self.conv2] + hlayers:
    #   normal_trunc(convlayer.weight)
    #   constant_test(convlayer.bias)

  def forward(self, *xs):
    assert len(xs) == len(self.in_sizes), "Wrong # inputs"
    x = self.combine_inputs(xs)
    x = self.conv1(x)
    if self.batch_norm:
      x = self.firstblayer(x)
    # x = self.activation(x)

    # h layers
    for (i, layer) in enumerate(self.hlayers):
      x = layer(x)
      if self.batch_norm:
        x = self.hblayers[i](x)
      x = self.activation(x)

    x = self.conv2(x)
    if self.batch_norm:
      x = self.lastblayer(x)
    x = self.activation(x)

    # Uncombine
    split_channels = split_channel(x, self.out_sizes)
    return tuple(map(slither, split_channels, self.out_sizes)) # FIXME, not all men
