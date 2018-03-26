import torch.nn as nn
from asl.archs.packing import nelem, ndims
import asl

class LinearNormalizeSize(nn.Module):

  def __init__(self, input_size, normalized_size):
    super(LinearNormalizeSize, self).__init__()
    self.normalized_size = normalized_size
    self.linear = nn.Linear(nelem(input_size),
                            nelem(normalized_size))

  def forward(self, x):
    return self.linear(x).view(asl.util.addbatchdim(self.normalized_size))

class IdentityNormalizeSize(nn.Module):
  def __init__(self, input_size, normalized_size):
    nn.Module.__init__(self)
    # super(IdentityNormalizeSize, self).__init__()

  def forward(self, x):
    return x

def normalizer(size, normalized_size):
  if size == normalized_size:  # do nothing
    return IdentityNormalizeSize
  elif ndims(size):
    return LinearNormalizeSize
  else:
    raise Exception

class NormalizeNet(nn.Module):
  def __init__(self,
               in_sizes,
               out_sizes):
    "Combines multiple inputs into one input"
    super(NormalizeNet, self).__init__()
    normalized_size = max(in_sizes, key=asl.archs.nelem)
    normalizers = [asl.archs.normalizer(size, normalized_size)(size, normalized_size)for size in in_sizes]
    self.size_normalizers = nn.ModuleList(normalizers)
  
  def forward(self, *xs):
    return [self.size_normalizers[i](x) for i, x in enumerate(xs)]