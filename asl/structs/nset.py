"Stack Data Structure trained from a reference implementation"
import asl
from asl.type import Function
from asl.modules.modules import ConstantNet, ModuleDict
from asl.archs.convnet import ConvNet
from asl.util.misc import cuda
from torch import nn

class Add(asl.Function, asl.Net):
  def __init__(self, SetType, ItemType, name="Add",  **kwargs):
    asl.Function.__init__(self, [SetType, ItemType], [SetType])
    asl.Net.__init__(self, name, **kwargs)

class Card(asl.Function, asl.Net):
  def __init__(self, SetType, IntegerType, name="Card", **kwargs):
    asl.Function.__init__(self, [SetType], [IntegerType])
    asl.Net.__init__(self, name, **kwargs)

class IsIn(asl.Function, asl.Net):
  "Is Item in the set"
  def __init__(self, SetType, ItemType, BoolType, name="IsIn", **kwargs):
    asl.Function.__init__(self, [SetType, ItemType], [BoolType])
    asl.Net.__init__(self, name, **kwargs)

class Union(asl.Function, asl.Net):
  def __init__(self, SetType, name="Union", **kwargs):
    asl.Function.__init__(self, [SetType, SetType], [SetType])
    asl.Net.__init__(self, name, **kwargs)

class Intersection(asl.Function, asl.Net):
  def __init__(self, SetType, name="Intersection", **kwargs):
    asl.Function.__init__(self, [SetType, SetType], [SetType])
    asl.Net.__init__(self, name, **kwargs)

def py_card(aset):
  "Cardinality of aset"
  return (len(aset),)


def py_in(aset, item):
  "Is item in a set"
  return (item in aset,)


def py_add(aset, item):
  aset = aset.copy()
  aset.add(item)
  return (aset,)

py_empty_set = set()