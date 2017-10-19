from enum import Enum
import torch.nn as nn
from torch.autograd import Variable
from asl.util.misc import cuda
from multipledispatch import dispatch
from asl.util.torch import onehot

"Common encodings"
# One Hot Encodings
class Encoding():
  def __init__(self, value):
    self.value = value

  def size(self):
    return self.value.size()


class OneHot1D(Encoding):
  pass


@dispatch(OneHot1D, OneHot1D)
def dist(x, y):
  # Check length
  return nn.BCEWithLogitsLoss()(x, y)


def compound_encoding(cl, encoding):
  "Class that is ClassEncoding"
  return next(x for x in cl.__subclasses__() if issubclass(x, encoding))


@dispatch(Enum)
def onehot1d(enum, length=None):
  "Encode a Color as a one hot vector"
  EnumOneHot1D = compound_encoding(enum.__class__.__bases__[0], OneHot1D)
  length = len(enum.__class__) if length is None else length
  return EnumOneHot1D(Variable(cuda(onehot(enum.value, length, 1))))

# @dispatch(ColorOneHot1D, ColorOneHot1D)
# def equal(x, y):
#   return nn.BCELoss()(x.value, y.value)
#
# @dispatch(Enum)
# def onehot1d(color):
#   "Encode a Color as a one hot vector"
#   return ColorOneHot1D(Variable(cuda(onehot(color.value, 8, 1))))
