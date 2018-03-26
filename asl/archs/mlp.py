import math
import random
from asl.archs.packing import split_channel, nelements, splt_reshape_tensors
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

class MLPNet(nn.Module):
  "ConvNet which takes variable inputs and variable outputs"

  def sample_hyper(in_sizes,
                   out_sizes,
                   pbatch_norm=0.5,
                   max_layers=5,
                   pact_same=0.5,
                   **kwargs):
    "Sample hyper parameters of MLP"
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

  def sample_hyper(in_sizes,
                   out_sizes,
                   pbatch_norm=0.5,
                   max_layers=5,
                   pact_same=0.5,
                   **kwargs):
    "Sample hyper parameters of MLP"
    batch_norm = np.random.rand() > pbatch_norm
    return {'batch_norm': batch_norm}


  def __init__(self,
               in_sizes,
               out_sizes,
               batch_norm=True,
               nmids=None,
               activations=None,
               output_act=lambda x: x):
    super(MLPNet, self).__init__()
    # import pdb; pdb.set_trace()
    self.nin = nelements(in_sizes)
    self.nout = nelements(out_sizes)
    nmids = [] if nmids is None else nmids
    self.in_sizes = in_sizes
    self.out_sizes = out_sizes
    self.batch_norm = batch_norm
    self.output_act = output_act
    nhlayers = len(nmids)

    # Layers
    layers = []
    bnlayers = []
    l2 = self.nout if nhlayers == 0 else nmids[0]
    layers.append(nn.Linear(self.nin, l2))
    if batch_norm:
      bnlayers.append(nn.BatchNorm1d(l2))

    if nhlayers > 0:
      layer_lens = nmids + [self.nout]
      self.layer_lens = layer_lens
      for i in range(len(layer_lens) - 1):
        layers.append(nn.Linear(layer_lens[i], layer_lens[i + 1]))
        if batch_norm:
          bnlayers.append(nn.BatchNorm1d(layer_lens[i + 1]))

    if batch_norm:
      self.bnlayers = nn.ModuleList(bnlayers)

    self.layers = nn.ModuleList(layers)
    if activations is None:
      self.activations = [F.elu for i in range(nhlayers)]
    else:
      self.activations = activations

  def forward(self, *xs):
    exp_xs = expand_consts(xs) # TODO: Make optional
    exp_xs = [x.contiguous().view(x.size(0), -1) for x in exp_xs]
    # Combine inputs
    x = torch.cat(exp_xs, dim=1)
    for (i, layer) in enumerate(self.layers):
      x = layer(x)
      if self.batch_norm:
        x = self.bnlayers[i](x)
      
      if i == len(self.layers) - 1:
        x = self.output_act(x)
      else:
        x = self.activations[i](x)

    # Uncombine inputs
    outxs = splt_reshape_tensors(x, self.out_sizes)
    res = [x.contiguous().view(x.size(0), *self.out_sizes[i]) for (i, x) in enumerate(outxs)]
    return res
