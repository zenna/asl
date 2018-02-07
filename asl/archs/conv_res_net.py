"Convolutional Residual Network"
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from asl.archs.packing import split_channel, cat_channels, slither
from asl.modules.modules import expand_consts
from .convnet import channels
import math

def interpolate_inc(a, b, npoints):
  "Integer interpolation between a and b inclusive"
  res = []
  x = a
  for _ in range(npoints):
    delta = (b - a) / (npoints + 1)
    x = x + delta
    res.append(math.floor(x))

  return [a] + res + [b]

class ConvActBn(nn.Module):
  def __init__(self,
               in_channels=8,
               out_channels=8,
               ks=3,
               activation=F.elu,
               block_size=2,
               padding=1,
               batch_norm=True,
               learn_batch_norm=True,
               bias=True,
               act=True):
    super(ConvActBn, self).__init__()
    self.batch_norm = batch_norm
    self.conv = nn.Conv2d(in_channels, out_channels, ks, padding=padding,
                          bias=bias)
    self.act = act
    self.act = activation
    if batch_norm:
      self.bn = nn.BatchNorm2d(out_channels, affine=False)
      if not learn_batch_norm:
        self.bn.eval() # Turn to eval mode to not learn parameters

  def forward(self, x):
    x = self.conv(x)
    if self.batch_norm:
      x = self.bn(x)
    if self.act:
      x = self.act(x)
    return x

class BasicBlock(nn.Module):
  def __init__(self, block_size, in_channels, out_channels, activation, **kwargs):
    super(BasicBlock, self).__init__()
    self.activation = activation
    channels = interpolate_inc(in_channels, out_channels, block_size - 1)
    assert len(channels) - 1 == block_size
    convs = []
    for i in range(block_size):
      a = channels[i]
      b = channels[i+1]
      act = i < block_size - 1
      conv = ConvActBn(in_channels=a, out_channels=b, act=act,
                       activation=activation, **kwargs)
      convs.append(conv)
    self.convs = nn.ModuleList(convs)
    self.project_resdiual = in_channels != out_channels
    if self.project_resdiual:
      self.projection = ConvActBn(in_channels=in_channels,
                                  out_channels=out_channels,
                                  ks=1,
                                  bias=False,
                                  padding=0)

  def forward(self, x):
    residual = x
    if self.project_resdiual:
      residual = self.projection(x)

    for conv in self.convs:
      x = conv(x)
    x = x + residual
    x = self.activation(x)
    return x

class ConvResNet(nn.Module):
  "ConvNet which takes variable inputs and variable outputs"

  def sample_hyper(in_sizes, out_sizes, pbatch_norm=0.5, max_layers=8):
    "Sample hyper parameters"
    ks = random.choice([3, 5])
    return {'batch_norm': np.random.rand() > pbatch_norm,
            'h_channels': random.choice([4, 8, 12, 16, 24]),
            'activation': random.choice([F.elu]),
            'ks': ks,
            'last_activation': random.choice([F.elu]),
            'learn_batch_norm': np.random.rand() > 0.5,
            'padding': (ks - 1)//2}

  def __init__(self,
               in_sizes,
               out_sizes,
               block_size=2,
               batch_norm=False,
               h_channels=8,
               ks=3,
               combine_inputs=cat_channels,
               activation=F.elu,
               last_activation=F.elu,
               learn_batch_norm=True,
               padding=1,
               nblocks=2,
               conv_init=nn.init.xavier_uniform):
    super(ConvResNet, self).__init__()
    import pdb; pdb.set_trace()
    # Assumes batch not in size and all in/out same size except channel
    self.in_sizes = in_sizes
    self.out_sizes = out_sizes
    self.activation = activation
    self.last_activation = last_activation
    self.combine_inputs = combine_inputs
    self.batch_norm = batch_norm
    in_channels = channels(in_sizes)
    out_channels = channels(out_sizes)

    blocks = []
    for i in range(nblocks):
      if i == 0:
        a = in_channels
        b = h_channels
      elif i == nblocks - 1:
        a = h_channels
        b = out_channels
      else:
        a = h_channels
        b = h_channels

      blocks.append(BasicBlock(block_size,
                               in_channels=a,
                               out_channels=b,
                               ks=ks,
                               activation=activation, 
                               padding=padding,
                               batch_norm=batch_norm,
                               learn_batch_norm=learn_batch_norm))
                    
    self.blocks = nn.ModuleList(blocks)

  def forward(self, *xs):
    assert len(xs) == len(self.in_sizes), "Wrong # inputs"
    x = self.combine_inputs(xs)
    for block in self.blocks:
      x = block(x)

    # Uncombine
    split_channels = split_channel(x, self.out_sizes)
    return tuple(map(slither, split_channels, self.out_sizes)) # FIXME, not all men
