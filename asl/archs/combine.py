import asl
from torch import nn
import torch

def cat_channels(xs, channel_dim=1):
  "Concatenate images in the channel dimension"
  return torch.cat(xs, dim=channel_dim)

class CombineNet(nn.Module):
  def __init__(self,
               combine_inputs = cat_channels):
    "Combines multiple inputs into one input"
    super(CombineNet, self).__init__()
    self.combine_inputs - combine_inputs

  def forward(self, *xs):
    return self.combine_inputs(xs)
