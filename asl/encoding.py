from enum import Enum
from multipledispatch import dispatch
import torch.nn as nn
from torch.autograd import Variable
from asl.util.misc import cuda
from asl.util.torch import onehot
from asl.type import Type
import torch

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
def equal(x, y):
  same = torch.max(x.value, 1)[1] == torch.max(y.value, 1)[1]
  return same.data[0]

@dispatch(OneHot1D, OneHot1D)
def dist(x, y):
  # Check length
  return nn.BCEWithLogitsLoss()(x.value, y.value)


def compound_encoding(cl, encoding):
  "Class that is ClassEncoding"
  return next(x for x in cl.__subclasses__() if issubclass(x, encoding))


@dispatch(Enum)
def onehot1d(enum, length=None):
  "Encode a Color as a one hot vector"
  EnumOneHot1D = compound_encoding(enum.__class__.__bases__[0], OneHot1D)
  length = EnumOneHot1D.typesize[0] if length is None else length
  return EnumOneHot1D(Variable(cuda(onehot(enum.value, length, 1))))
