import torch.nn as nn


class Type:
  """A type is a Type"""
  def __init__(self, name, size, dtype):
    self.name = name
    self.size = size
    self.dtype = dtype

  def size(self):
    return self.size


class FunctionType:
  "Function Type"

  def __init__(self, in_types, out_types):
    self.in_types = in_types
    self.out_types = out_types


class Function():
  "Typed Function"

  def __init__(self):
    super(Function, self).__init__()

class Constant:
  "Typed Constant"

  def __init__(self, type, value):
    self.type = type
    self.value = value

  def parameters(self):
    return [self.value]

  def __call__(self):
    return self.value.expand(32, 1, 28, 28)
