"Templates (modules parameterized by shape)"
from asl.modules.modules import expand_consts, ModuleDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from asl.util.misc import mul_product

class MyClass(nn.Module):
  def __init__(self):
    super(MyClass, self).__init__()
    # self.bok = nn.ModuleList([nn.Conv2d(10, 10, 3, padding=1) for i in range(3)])
    self.mda = ModuleDict({str(i): nn.Conv2d(10, 10, 3, padding=1) for i in range(3)})



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

  def __init__(self, in_sizes, out_sizes,
               channel_dim=1,
               batch_norm=False,
               h_channels=16,
               nhlayers=24,
               activation=F.elu):
    import pdb; pdb.set_trace()
    super(VarConvNet, self).__init__()
    import pdb; pdb.set_trace()
    # Assumes batch not in size and all in/out same size except channel
    self.in_sizes = in_sizes
    self.out_sizes = out_sizes
    self.channel_dim = channel_dim
    self.activation = activation
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
    x = torch.cat(exp_xs, dim=self.channel_dim)
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


class MLPNet(nn.Module):
  "ConvNet which takes variable inputs and variable outputs"

  def __init__(self, in_sizes, out_sizes, channel_dim=1):
    super(MLPNet, self).__init__()
    # Assumes batch not in size and all in/out same size except channel
    self.flat_in_size = [mul_product(size) for size in in_sizes]
    self.flat_out_size = [mul_product(size) for size in out_sizes]

    self.nin = sum(self.flat_in_size)
    self.nout = sum(self.flat_out_size)
    self.nmid = math.floor((self.nin + self.nout) / 2)
    self.in_sizes = in_sizes
    self.out_sizes = out_sizes
    self.m1 = nn.Linear(self.nin, self.nmid)
    self.m2 = nn.Linear(self.nmid, self.nout)

  def forward(self, *xs):
    assert len(xs) == len(self.in_sizes), "Wrong # inputs"
    exp_xs = expand_consts(xs) # TODO: Make optional
    exp_xs = [x.contiguous().view(x.size(0), -1) for x in exp_xs]
    x = torch.cat(exp_xs, dim=1)
    x = self.m1(x)
    x = F.elu(x)
    y = self.m2(x)

    outxs = unstack_channel(y, self.out_sizes)
    res = [x.contiguous().view(x.size(0), *self.out_sizes[i]) for (i, x) in enumerate(outxs)]
    return res
