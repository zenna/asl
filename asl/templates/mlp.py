import math
from asl.templates.packing import split_channel
from asl.util.misc import mul_product
from asl.modules.modules import expand_consts
import torch
import torch.nn.functional as F
import torch.nn as nn


class MLPNet(nn.Module):
  "ConvNet which takes variable inputs and variable outputs"

  def __init__(self,
               in_sizes,
               out_sizes,
               channel_dim=1,
               nblocks=1,
               block_size=2):
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
    import pdb; pdb.set_trace()
    exp_xs = expand_consts(xs) # TODO: Make optional
    exp_xs = [x.contiguous().view(x.size(0), -1) for x in exp_xs]
    x = torch.cat(exp_xs, dim=1)
    x = self.m1(x)
    x = F.elu(x)
    y = self.m2(x)
    y = F.sigmoid(y)

    outxs = split_channel(y, self.out_sizes)
    res = [x.contiguous().view(x.size(0), *self.out_sizes[i]) for (i, x) in enumerate(outxs)]
    return res
