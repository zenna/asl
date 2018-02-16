import asl
from torch import nn

class CombineNet(nn.Module):
  def __init__(self,
               *,
               in_sizes,
               out_sizes):
    "Combines multiple inputs into one input"
    super(CombineNet, self).__init__()
    normalized_size = max(in_sizes, key=asl.archs.nelem)
    normalizers = [asl.archs.normalizer(size, normalized_size)(size, normalized_size)for size in in_sizes]
    self.size_normalizers = nn.ModuleList(normalizers)
  
  def forward(self, *xs):
    xs_same_size = [self.size_normalizers[i](x) for i, x in enumerate(xs)]
    x = self.combine_inputs(xs_same_size)
