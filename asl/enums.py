from enum import Enum
from multipledispatch import dispatch
from asl.util.torch import onehot
from torch.autograd import Variable
from asl.util.misc import cuda
import torch.nn as nn
from asl.encoding import OneHot1D, OneHot2D, onehot1d
import inspect

class Color():
  pass

# But then if im just passing around one hot then hwo can I
class ColorEnum(Color, Enum):
  red = 0
  green = 1
  gray = 2
  yellow = 3
  blue = 4
  cyan = 5
  brown = 6
  purple = 7


class Material():
  pass


class MaterialEnum(Enum):
  metal = 0
  rubber = 1


class ColorOneHot1D(Color, OneHot1D):
  pass

class MaterialOneHot1D(Material, OneHot1D):
  pass


class NeuralColor(Color):
  def __init__(self, data):
    self.value = value
