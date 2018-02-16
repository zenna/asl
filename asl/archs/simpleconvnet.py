"Templates (modules parameterized by shape)"
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from asl.archs.packing import split_channel, cat_channels, slither
from asl.modules.modules import expand_consts
import asl

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


class SimpleConvNet(nn.Module):
  "ConvNet -> BatchNorm -> Activation layers"

  @staticmethod
  def sample_hyper(in_sizes, out_sizes, pbatch_norm=0.5, max_layers=8):
    "Sample hyper parameters"
    batch_norm = np.random.rand() > pbatch_norm
    learn_batch_norm = np.random.rand() > 0.5
    nlayers = np.random.randint(0, max_layers)
    h_channels = random.choice([4, 8, 12, 16, 24])
    act = random.choice([F.elu])
    ks = random.choice([3, 5])
    conv_init = random.choice([nn.init.xavier_uniform])
    return {'batch_norm': batch_norm,
            'h_channels': h_channels,
            'nhlayers': nlayers,
            'activation': act,
            'ks': ks,
            'learn_batch_norm': learn_batch_norm,
            'padding': (ks - 1)//2,
            'conv_init': conv_init}

  @staticmethod
  def args_from_sizes(in_sizes, out_sizes):
    in_sizes=None
    out_sizes=None
    in_channels = channels(in_sizes)
    out_channels = channels(out_sizes)
    return {"in_channels": in_channels, "out_channels": out_channels}

  def __init__(self,
               *,
               in_channels,
               out_channels,
               h_channels=8,
               nhlayers=4,
               ks=3,
               activation=F.elu,
               batch_norm=False,
               learn_batch_norm=True,
               batch_norm_last=False,
               padding=1,
               conv_init=nn.init.xavier_uniform):
    super(SimpleConvNet, self).__init__()
    self.activation = activation
    self.batch_norm = batch_norm
    self.batch_norm_last = batch_norm_last

    nchans = [in_channels] + [h_channels for _ in range(nhlayers)] + [out_channels]
    nlayers = nhlayers + 2 # add in/out channels
    layers = [nn.Conv2d(nchans[i], nchans[i+1], ks, padding=padding)
              for i in nlayers - 1]

    # Batch norm
    if batch_norm:
      n = len(nchans) if batch_norm_last else len(nchans) - 1
      allblayers = [nn.BatchNorm2d(nchans[i]) for i in range(1, n)]
      self.allblayers = nn.ModuleList(allblayers)
      if not learn_batch_norm:
        self.allblayers.eval() # Turn to eval mode to not learn parameters

    # Init Layers
    for convlayer in layers:
      conv_init(convlayer.weight)

  def forward(self, x):
    for (i, layer) in enumerate(self.hlayers):
      x = layer(x)
      if self.batch_norm:
        if i != final or self.batch_norm_last:
          x = self.hblayers[i](x)
      x = self.activation(x)
    
    return x