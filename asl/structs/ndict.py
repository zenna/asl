"Stack Data Structure trained from a reference implementation"
from asl.type import Function
from asl.modules.modules import ConstantNet, ModuleDict
from asl.archs.convnet import ConvNet
from asl.util.misc import cuda
from torch import nn

class SetItem(Function):
  "SetItem Function for Stack"

  def __init__(self, dict_type, key_type, value_type):
    super(SetItem, self).__init__([dict_type, key_type, value_type],
                                  [dict_type])


class GetItem(Function):
  "GetItem Function for Stack"

  def __init__(self, dict_type, key_type, value_type):
    super(GetItem, self).__init__([dict_type, key_type], [value_type])

def dict_set_item(adict, key, value):
  adict = adict.copy()
  adict[key] = value
  return (adict, )


def dict_get_item(adict, key):
  return (adict[key], )

dict_empty = {}

def ref_dict():
  return {"set_item": dict_set_item, "get_item": dict_get_item, "empty": {}}
