"Stack Data Structure trained from a reference implementation"
from asl.type import Function
from asl.modules.modules import ConstantNet, ModuleDict
from asl.templates.convnet import ConvNet
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


class SetItemNet(SetItem, nn.Module):
  def __init__(self, dict_type, key_type, value_type, module=None, template=ConvNet,
               template_opt=None):
    super(SetItemNet, self).__init__(dict_type, key_type, value_type)
    if module is None:
      template_opt = {} if template_opt is None else template_opt
      self.module = template(self.in_sizes(), self.out_sizes(), **template_opt)
    else:
      self.module = module
    self.add_module("SetItem", self.module)

  def forward(self, *xs):
    return self.module.forward(*xs)


class GetItemNet(GetItem, nn.Module):
  def __init__(self, dict_type, key_type, value_type, module=None, template=ConvNet,
               template_opt=None):
    super(GetItemNet, self).__init__(dict_type, key_type, value_type)
    if module is None:
      template_opt = {} if template_opt is None else template_opt
      self.module = template(self.in_sizes(), self.out_sizes(), **template_opt)
    else:
      self.module = module
    self.add_module("GetItem", self.module)


  def forward(self, *xs):
    return self.module.forward(*xs)


def dict_set_item(adict, key, value):
  adict = adict.copy()
  adict[key] = value
  return (adict, )


def dict_get_item(adict, key):
  return (adict[key], )


def ref_dict():
  return {"set_item": dict_set_item, "get_item": dict_get_item, "empty": {}}
