"Stack Data Structure trained from a reference implementation"
from asl.type import Function
from asl.modules.modules import ConstantNet, ModuleDict
from asl.archs.convnet import ConvNet
from asl.util.misc import cuda
import asl

class GetItem(asl.Function, asl.Net):
  def __init__(self, DictType, KeyType, ValueType, name="GetItem", **kwargs):
    asl.Function.__init__(self, [DictType, KeyType], [ValueType])
    asl.Net.__init__(self, name, **kwargs)

class SetItem(asl.Function, asl.Net):
  def __init__(self, DictType, KeyType, ValueType, name="SetItem", **kwargs):
    asl.Function.__init__(self, [DictType, KeyType, ValueType], [DictType])
    asl.Net.__init__(self, name, **kwargs)

def dict_set_item(adict, key, value):
  adict = adict.copy()
  adict[key] = value
  return (adict, )

def dict_get_item(adict, key):
  return (adict[key], )

dict_empty = {}