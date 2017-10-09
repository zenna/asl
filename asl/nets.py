import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class VarConvNet(nn.Module):
  "ConvNet which takes variable inputs and variable outputs"

  def __init__(self, in_shapes, out_shapes):
    super(VarConvNet, self).__init__()
    in_channels = stack_channels + img_channels
    out_channels = stack_channels
    nf = 16
    self.conv1 = nn.Conv2d(in_channels, nf, 3, padding=1)
    self.convmid = nn.Conv2d(nf, nf, 3, padding=1)
    self.conv2 = nn.Conv2d(nf, stack_channels, 3, padding=1)

  def forward(self, x, y):
    x = torch.cat([x, y], dim=1)
    # import pdb; pdb.set_trace()
    x = F.elu(self.conv1(x))
    x = F.elu(self.conv2(x))
    return (x,)
