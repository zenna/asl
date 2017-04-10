"""Types"""
from typing import Sequence, Callable
from wacacore.util.misc import getn
import numpy as np
from collections import namedtuple

# Spec
# ====

# - Be able to do this symbolic transformation
# - Be able to easily change the discrete parts of the data-distribution
# each training step
# - Be able to add arbitrary other loss terms to the data-distribution
# - Dynamic shape?
# - more like julia style, have types and functions kinda separate
# - higher order?

# Types
# =====


class Type():
  """A Data Type, i.e. a name distinguishd class of things."""

  def __init__(self, name: str):
    # TODO: Make exception not assertion
    assert name[0].isupper(), "By convention, first letter of type is upper"
    self.name = name

  def __str__(self):
    return self.name

  def __repr__(self):
    return str(self)


class TupleType():
  """Type of a tuple/product: (T1, T2, T3)"""
  def __init__(self, types: Sequence[Type]):
    self._types = tuple(types)

  def __getitem__(self, i: int):
    return self._types[i]

  def __str__(self):
    return "({})".format("×".join([str(type) for type in self._types]))

  def __repr__(self):
    return str(self)

  def __len__(self):
    return len(self._types)


class FunctionType():
  """Type of a Function"""
  def __init__(self,
               lhs_types: Sequence[Type],
               rhs_types: Sequence[Type]):
      if len(lhs_types) < 1 or len(rhs_types) < 1:
        raise ValueError("Function input and output types must > 1")
      self.lhs_types = TupleType(lhs_types)
      self.rhs_types = TupleType(rhs_types)

  def __str__(self):
    return "{} -> {}".format(self.lhs_types, self.rhs_types)

  def __repr__(self):
    return str(self)


# Variables
# =========
class Variable():
  """A Variable varies over things"""
  pass


class Function(Variable):
  """A Function transforms one or more lhs_types to one or more rhs_types"""
  def __init__(self,
               lhs_types: Sequence[Type],
               rhs_types: Sequence[Type],
               name: str):
    self._type = FunctionType(lhs_types, rhs_types)
    self.name = name

  def type(self):
    """Type of a function variable"""
    return self._type

  def __call__(self, *lhs_vars: Sequence[Variable]):
    if not types_vars_consistent(self._type.lhs_types._types, lhs_vars):
      lhs_types = self.type().lhs_types
      lhs_var_types = tuple([v.type() for v in lhs_vars])
      error_msg = """{} expects {} arguments of type {},
        got {} arguments of type {}""".format(self.name,
                                              len(lhs_types),
                                              lhs_types,
                                              len(lhs_var_types),
                                              lhs_var_types)
      raise TypeError(error_msg)
      # TODO: represent output numbers
      # TODO: represent or be able to infer its type
    return FunctionApp(self, lhs_vars)

  def __str__(self):
    return self.name

  def __repr__(self):
    return str(self)


class Constant(Variable):
  """A Constant of a particular `Type`"""

  def __init__(self,
               type: Type,
               name: str):
    if not str.isupper(name):
      raise ValueError("Constant names by convention upper case")
    self._type = type
    self.name = name

  def type(self):
    """Return type of constant"""
    return self._type

  def __str__(self):
    return "{}:{}".format(self.name, self.type())

  def __repr__(self):
    return str(self)


class ForAllVar(Variable):
  "A (symbolic) universally quantified variable"

  def __init__(self, type: Type, name: str):
    self._type = type
    self.name = name

  def type(self) -> Type:
    """`Type` of a variable"""
    return self._type

  def __str__(self):
    return "{}:{}".format(self.name, self._type)

  def __repr__(self):
    return str(self)


class IndexApp(Variable):
  """Indexing of a tuple"""

  def __init__(self, var, i: int):
    self.var = var
    self.index = i

  def type(self):
    return self.var.type()[self.index]

  def __str__(self):
    return "{}[{}]:{}".format(self.var, self.index, self.type())

  def __repr__(self):
    return str(self)

class FunctionApp(Variable):
  """Function application"""

  def __init__(self, function: Function, lhs_vars: Sequence[Variable]):
    # TODO: type_check
    self.function = function
    self.lhs_vars = lhs_vars

  def __getitem__(self, i: int):
    assert 0 <= i < len(self.type())
    return IndexApp(self, i)

  def type(self):
    return self.function._type.rhs_types

  def __str__(self):
    arg_strings = ", ".join([str(var) for var in self.lhs_vars])
    return "{}({}):{}".format(self.function, arg_strings, self.type())

  def __repr__(self):
    return str(self)

def type_type_consistent(type_a: Type, type_b: Type) -> bool:
  """Is type_a consistent with type_b"""
  return type_a == type_b

def type_var_consistent(type: Type, var:Variable) -> bool:
  return type_type_consistent(type, var.type())

def types_vars_consistent(types: Sequence[Type],
                          vars:Sequence[Variable]) -> bool:
  """Are the types of `vars` consistent with `types`"""
  if len(types) != len(vars):
    return False
  else:
    return all((type_var_consistent(types[i], vars[i]) \
               for i in range(len(types))))


# Constraints
# ===========
class EqAxiom():
  """An equational axiom `lhs == rhs`"""
  def __init__(self,
               lhs: Variable,
               rhs: Variable,
               name: str):
    self.lhs = lhs
    self.rhs = rhs
    self.name = name

# Concrete Methods
# ================

class Method():
  """A method is an implementation of abstract function"""

  def __init__(self, type: FunctionType, call: Callable):
    self.call = call
    self.type = type

  def __call__(self, *args):
    return self.call(*args)
