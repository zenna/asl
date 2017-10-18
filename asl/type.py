class Type(object):
  """A type is a Type"""
  def __init__(self, name, size, dtype, observable=True):
    self.name = name
    self.size = size
    self.dtype = dtype
    self.observable = observable

  def size():
    return _size


class FunctionType(object):
  "Function Type"

  def __init__(self, in_types, out_types):
    self.in_types = in_types
    self.out_types = out_types


class Function(object):
  "Typed Function"

  def __init__(self, in_types, out_types):
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


class Constant(object):
  "Typed Constant"

  def __init__(self, type):
    self.type = type

  def size(self):
    return self.type.size
