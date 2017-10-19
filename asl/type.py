class Type(object):
  def __init__(self, value):
    self.value = value

  @classmethod
  def observable(cls):
    return cls.observable

  def size(self):
    return self.value.size()


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
    self.func_type = FunctionType(self.in_types, self.out_types)

  def in_sizes(self):
    return [type.typesize for type in self.func_type.in_types]

  def out_sizes(self):
    return [type.typesize for type in self.func_type.out_types]

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
