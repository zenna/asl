
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

  def __init__(self, in_types, out_types):
    super(Function, self).__init__()
    self.in_types = in_types
    self.out_types = out_types

  def type(self):
    "Function Type"
    return FunctionType(self.in_types, self.out_types)

  def in_sizes(self):
    return [type.size for type in self.type().in_types]

  def out_sizes(self):
    return [type.size for type in self.type().out_types]

  def n_inputs(self):
    return len(self.type().in_types)

  def n_outputs(self):
    return len(self.type().out_types)




class Constant:
  "Typed Constant"

  def __init__(self, type, value, batch_size):
    self.type = type
    self.value = value
    self.batch_size = batch_size
    self.expanded_size = (batch_size, ) + type.size

  def parameters(self):
    return [self.value]

  def __call__(self):
    return self.value.expand(*self.expanded_size)
