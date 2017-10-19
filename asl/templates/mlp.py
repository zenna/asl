import math
import random
from asl.templates.packing import split_channel
from asl.util.misc import mul_product
from asl.modules.modules import expand_consts
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

def interpolate(a, b, npoints):
  res = []
  x = a
  for _ in range(npoints):
    delta = (b - a) / (npoints + 1)
    x = x + delta
    res.append(math.floor(x))

  return res


def nelements(sizes):
  "Total number for elements from set of sizes"
  size = [mul_product(size) for size in sizes]
  return sum(size)


class MLPNet(nn.Module):
  "ConvNet which takes variable inputs and variable outputs"

  def sample_hyper(in_sizes,
                   out_sizes,
                   pbatch_norm=0.5,
                   max_layers=5,
                   pact_same=0.5):
    "Sample hyper parameters"
    nin = nelements(in_sizes)
    nout = nelements(out_sizes)
    batch_norm = np.random.rand() > pbatch_norm
    nhlayers = np.random.randint(0, max_layers)
    nmids = interpolate(nin, nout, nhlayers)
    mutliplier = random.choice([1, 2])
    nmids = [nm * mutliplier for nm in nmids]
    same_act = np.random.rand() > pact_same
    nacts = nhlayers + 1
    if same_act:
      act = random.choice([F.relu, F.elu])
      activations = [act for i in range(nacts)]
    else:
      activations = [random.choice([F.relu, F.elu]) for i in range(nacts)]

    return {'batch_norm': batch_norm,
            'activations': activations,
            'nmids': nmids}


  def __init__(self,
               in_sizes,
               out_sizes,
               batch_norm=True,
               nmids=None,
               activations=None):
    super(MLPNet, self).__init__()
    nhlayers = len(nmids)
    self.nin = nelements(in_sizes)
    self.nout = nelements(out_sizes)
    nmids = [] if nmids is None else nmids
    self.in_sizes = in_sizes
    self.out_sizes = out_sizes

    # Layers
    layers = []
    l2 = self.nout if nhlayers == 0 else nmids[0]
    layers.append(nn.Linear(self.nin, l2))
    if nhlayers > 0:
      layer_lens = nmids + [self.nout]
      self.layer_lens = layer_lens
      for i in range(len(layer_lens) - 1):
        layers.append(nn.Linear(layer_lens[i], layer_lens[i + 1]))

    self.layers = nn.ModuleList(layers)
    if activations is None:
      self.activations = [F.elu for i in range(layers)]
    else:
      self.activations = activations

  def forward(self, *xs):
    exp_xs = expand_consts(xs) # TODO: Make optional
    exp_xs = [x.contiguous().view(x.size(0), -1) for x in exp_xs]
    # Combine inputs
    x = torch.cat(exp_xs, dim=1)
    for (i, layer) in enumerate(self.layers):
      x = layer(x)
      # if self.batch_norm:
      #   x = self.blayers[i](x)
      x = self.activations[i](x)

    # Uncombine inputs
    outxs = split_channel(x, self.out_sizes)
    res = [x.contiguous().view(x.size(0), *self.out_sizes[i]) for (i, x) in enumerate(outxs)]
    return res
