import os
import torchvision
import torchvision.transforms as transforms
import asl
from asl.util.io import datadir
import torch
from torch.autograd import Variable


def image_data(data, nocuda=False):
  "Extract image data from mnist (and not classification)"
  return asl.cuda(Variable(data[0]), nocuda)