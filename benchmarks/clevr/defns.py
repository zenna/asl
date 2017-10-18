from asl.type import Function, Type
import torch.nn as nn
from asl.templates.mlp import MLPNet

# right now the push abstraction is doing nothing useufl
# what might it do?
#

def type_check(xs, types):
  assert len(xs) == len(types)
  for i, x in enumerate(xs):
    same_size = xs[i].size()[1:] == types[i].size
    assert same_size
  return xs


class Net(nn.Module):
  def __init__(self, name, module=None, template=MLPNet, template_opt=None):
    super(Net, self).__init__()
    if module is None:
      template_opt = {} if template_opt is None else template_opt
      self.module = template(self.in_sizes(), self.out_sizes(), **template_opt)
    else:
      self.module = module
    self.add_module(name, self.module)

  def forward(self, *xs):
    args = type_check(xs, self.in_types)
    res = self.module.forward(*args)
    return type_check(res, self.out_types)

# Fixed types

class SceneGraph(Type):
  "Stack represented as a vector"
  size = (10, 2, 3)

class Object(Type):
  "Object represented as one hot matrix"
  size = (4, 8)

class ObjectSet(Type):
  "Object set represented as one hot matrix"
  size = (10, 4, 8)


class Relation(Type):
  "Object set represented as one hot matrix"
  size = (4, 10, 10)


class Boolean(Type):
  "Object set represented as one hot matrix"
  size = (1,)


class Integer(Type):
  "Object set represented as one hot matrix"
  size = (10,)


class Color(Type):
  "Object set represented as one hot matrix"
  size = (8,)


class Material(Type):
  "Object set represented as one hot matrix"
  size = (8,)


class Shape(Type):
  "Object set represented as one hot matrix"
  size = (8,)


class Size(Type):
  "Object set represented as one hot matrix"
  size = (8,)

class Scene(Function, Net):
  def __init__(self, name="Scene", module=None, template=MLPNet, template_opt=None):
    Function.__init__(self, [SceneGraph], [ObjectSet])
    Net.__init__(self, "Scene", module, template, template_opt)


class Unique(Function, Net):
  def __init__(self="Unique", name="Unique", module=None, template=MLPNet, template_opt=None):
    Function.__init__(self, [ObjectSet], [Object])
    Net.__init__(self, " name", module, template, template_opt)


class Relate(Function, Net):
  def __init__(self="Relate", name="Relate", module=None, template=MLPNet, template_opt=None):
    Function.__init__(self, [SceneGraph, Object, Relation], [ObjectSet])
    Net.__init__(self, " name", module, template, template_opt)


class Count(Function, Net):
  def __init__(self, name="Count", module=None, template=MLPNet, template_opt=None):
    Function.__init__(self, [ObjectSet], [Integer])
    Net.__init__(self, "Count", module, template, template_opt)


class Exist(Function, Net):
  def __init__(self, name="Exist", module=None, template=MLPNet, template_opt=None):
    Function.__init__(self, [ObjectSet], [Boolean])
    Net.__init__(self, "Exist", module, template, template_opt)


class FilterSize(Function, Net):
  def __init__(self, name="FilterSize", module=None, template=MLPNet, template_opt=None):
    Function.__init__(self, [ObjectSet, Size], [ObjectSet])
    Net.__init__(self, "FilterSize", module, template, template_opt)


class FilterColor(Function, Net):
  def __init__(self, name="FilterColor", module=None, template=MLPNet, template_opt=None):
    Function.__init__(self, [ObjectSet, Color], [ObjectSet])
    Net.__init__(self, "FilterColor", module, template, template_opt)


class FilterMaterial(Function, Net):
  def __init__(self, name="FilterMaterial", module=None, template=MLPNet, template_opt=None):
    Function.__init__(self, [ObjectSet, Material], [ObjectSet])
    Net.__init__(self, "FilterMaterial", module, template, template_opt)


class FilterShape(Function, Net):
  def __init__(self, name="FilterShape", module=None, template=MLPNet, template_opt=None):
    Function.__init__(self, [ObjectSet, Shape], [ObjectSet])
    Net.__init__(self, "FilterShape", module, template, template_opt)


class Intersect(Function, Net):
  def __init__(self, name="Intersect", module=None, template=MLPNet, template_opt=None):
    Function.__init__(self, [ObjectSet, ObjectSet], [ObjectSet])
    Net.__init__(self, "Intersect", module, template, template_opt)


class Union(Function, Net):
  def __init__(self, name="Union", module=None, template=MLPNet, template_opt=None):
    Function.__init__(self, [ObjectSet, ObjectSet], [ObjectSet])
    Net.__init__(self, "Union", module, template, template_opt)


class GreaterThan(Function, Net):
  def __init__(self, name="GreaterThan", module=None, template=MLPNet, template_opt=None):
    Function.__init__(self, [Integer, Integer], [Boolean])
    Net.__init__(self, "GreaterThan", module, template, template_opt)


class LessThan(Function, Net):
  def __init__(self, name="LessThan", module=None, template=MLPNet, template_opt=None):
    Function.__init__(self, [Integer, Integer], [Boolean])
    Net.__init__(self, "LessThan", module, template, template_opt)


class EqualInteger(Function, Net):
  def __init__(self, name="EqualInteger", module=None, template=MLPNet, template_opt=None):
    Function.__init__(self, [Integer, Integer], [Boolean])
    Net.__init__(self, "EqualInteger", module, template, template_opt)


class EqualMaterial(Function, Net):
  def __init__(self, name="EqualMaterial", module=None, template=MLPNet, template_opt=None):
    Function.__init__(self, [Material, Material], [Boolean])
    Net.__init__(self, "EqualMaterial", module, template, template_opt)


class EqualSize(Function, Net):
  def __init__(self, name="EqualSize", module=None, template=MLPNet, template_opt=None):
    Function.__init__(self, [Size, Size], [Boolean])
    Net.__init__(self, "EqualSize", module, template, template_opt)


class EqualShape(Function, Net):
  def __init__(self, name="EqualShape", module=None, template=MLPNet, template_opt=None):
    Function.__init__(self, [Shape, Shape], [Boolean])
    Net.__init__(self, "EqualShape", module, template, template_opt)


class EqualColor(Function, Net):
  def __init__(self, name="EqualColor", module=None, template=MLPNet, template_opt=None):
    Function.__init__(self, [Color, Color], [Boolean])
    Net.__init__(self, "EqualColor", module, template, template_opt)


class QueryShape(Function, Net):
  def __init__(self, name="QueryShape", module=None, template=MLPNet, template_opt=None):
    Function.__init__(self, [Object], [Shape])
    Net.__init__(self, "QueryShape", module, template, template_opt)


class QuerySize(Function, Net):
  def __init__(self, name="QuerySize", module=None, template=MLPNet, template_opt=None):
    Function.__init__(self, [Object], [Size])
    Net.__init__(self, "QuerySize", module, template, template_opt)


class QueryMaterial(Function, Net):
  def __init__(self, name="QueryMaterial", module=None, template=MLPNet, template_opt=None):
    Function.__init__(self, [Object], [Material])
    Net.__init__(self, "QueryMaterial", module, template, template_opt)


class QueryColor(Function, Net):
  def __init__(self, name="QueryColor", module=None, template=MLPNet, template_opt=None):
    Function.__init__(self, [Object], [Color])
    Net.__init__(self, "QueryColor", module, template, template_opt)


class SameShape(Function, Net):
  def __init__(self, name="SameShape", module=None, template=MLPNet, template_opt=None):
    Function.__init__(self, [SceneGraph, Object], [ObjectSet])
    Net.__init__(self, "SameShape", module, template, template_opt)


class SameSize(Function, Net):
  def __init__(self, name="SameSize", module=None, template=MLPNet, template_opt=None):
    Function.__init__(self, [SceneGraph, Object], [ObjectSet])
    Net.__init__(self, "SameSize", module, template, template_opt)


class SameMaterial(Function, Net):
  def __init__(self, name="SameMaterial", module=None, template=MLPNet, template_opt=None):
    Function.__init__(self, [SceneGraph, Object], [ObjectSet])
    Net.__init__(self, "SameMaterial", module, template, template_opt)


class SameColor(Function, Net):
  def __init__(self, name="SameColor", module=None, template=MLPNet, template_opt=None):
    Function.__init__(self, [SceneGraph, Object], [ObjectSet])
    Net.__init__(self, "SameColor", module, template, template_opt)
