from asl.type import Function, Type
import torch.nn as nn
from asl.archs.mlp import MLPNet
from asl.net import Net


class Relations(Type):
  "Stack represented as a vector"
  size = (4, 10, 10)


class Object(Type):
  "Object represented as one hot matrix"
  size = (4, 8)


class ObjectSet(Type):
  "Object set represented as one hot matrix"
  size = (10, 4, 8)


class Relation(Type):
  "Relation as sparse relation matrix"
  size = (8, )


class Boolean(Type):
  "Boolean"
  size = (2,)


class Integer(Type):
  "Integer"
  size = (11,)


class Color(Type):
  "Enum, one hot"
  size = (8,)


class Material(Type):
  "Maeterial One hot"
  size = (8,)


class Shape(Type):
  "Shape one hto"
  size = (8,)


class Size(Type):
  "Size one hot"
  size = (8,)


class Unique(Function, Net):
  def __init__(self="Unique", name="Unique", **kwargs):
    Function.__init__(self, [ObjectSet], [Object])
    Net.__init__(self, " name", **kwargs)


class Relate(Function, Net):
  def __init__(self="Relate", name="Relate", **kwargs):
    Function.__init__(self, [Relations, Object, Relation], [ObjectSet])
    Net.__init__(self, " name", **kwargs)


class Count(Function, Net):
  def __init__(self, name="Count", **kwargs):
    Function.__init__(self, [ObjectSet], [Integer])
    Net.__init__(self, "Count", **kwargs)


class Exist(Function, Net):
  def __init__(self, name="Exist", **kwargs):
    Function.__init__(self, [ObjectSet], [Boolean])
    Net.__init__(self, "Exist", **kwargs)


class Filter(Function, Net):
  def __init__(self, name="Filter", **kwargs):
    Function.__init__(self, [ObjectSet, Size], [ObjectSet])
    Net.__init__(self, "Filter", **kwargs)


class FilterSize(Function, Net):
  def __init__(self, name="FilterSize", **kwargs):
    Function.__init__(self, [ObjectSet, Size], [ObjectSet])
    Net.__init__(self, "FilterSize", **kwargs)


class FilterColor(Function, Net):
  def __init__(self, name="FilterColor", **kwargs):
    Function.__init__(self, [ObjectSet, Color], [ObjectSet])
    Net.__init__(self, "FilterColor", **kwargs)


class FilterMaterial(Function, Net):
  def __init__(self, name="FilterMaterial", **kwargs):
    Function.__init__(self, [ObjectSet, Material], [ObjectSet])
    Net.__init__(self, "FilterMaterial", **kwargs)


class FilterShape(Function, Net):
  def __init__(self, name="FilterShape", **kwargs):
    Function.__init__(self, [ObjectSet, Shape], [ObjectSet])
    Net.__init__(self, "FilterShape", **kwargs)


class Intersect(Function, Net):
  def __init__(self, name="Intersect", **kwargs):
    Function.__init__(self, [ObjectSet, ObjectSet], [ObjectSet])
    Net.__init__(self, "Intersect", **kwargs)


class Union(Function, Net):
  def __init__(self, name="Union", **kwargs):
    Function.__init__(self, [ObjectSet, ObjectSet], [ObjectSet])
    Net.__init__(self, "Union", **kwargs)


class GreaterThan(Function, Net):
  def __init__(self, name="GreaterThan", **kwargs):
    Function.__init__(self, [Integer, Integer], [Boolean])
    Net.__init__(self, "GreaterThan", **kwargs)


class LessThan(Function, Net):
  def __init__(self, name="LessThan", **kwargs):
    Function.__init__(self, [Integer, Integer], [Boolean])
    Net.__init__(self, "LessThan", **kwargs)


class EqualInteger(Function, Net):
  def __init__(self, name="EqualInteger", **kwargs):
    Function.__init__(self, [Integer, Integer], [Boolean])
    Net.__init__(self, "EqualInteger", **kwargs)


class Equal(Function, Net):
  def __init__(self, name="Equal", **kwargs):
    Function.__init__(self, [Material, Material], [Boolean])
    Net.__init__(self, "Equal", **kwargs)


class EqualMaterial(Function, Net):
  def __init__(self, name="EqualMaterial", **kwargs):
    Function.__init__(self, [Material, Material], [Boolean])
    Net.__init__(self, "EqualMaterial", **kwargs)


class EqualSize(Function, Net):
  def __init__(self, name="EqualSize", **kwargs):
    Function.__init__(self, [Size, Size], [Boolean])
    Net.__init__(self, "EqualSize", **kwargs)


class EqualShape(Function, Net):
  def __init__(self, name="EqualShape", **kwargs):
    Function.__init__(self, [Shape, Shape], [Boolean])
    Net.__init__(self, "EqualShape", **kwargs)


class EqualColor(Function, Net):
  def __init__(self, name="EqualColor", **kwargs):
    Function.__init__(self, [Color, Color], [Boolean])
    Net.__init__(self, "EqualColor", **kwargs)


class Query(Function, Net):
  def __init__(self, name="Query", **kwargs):
    Function.__init__(self, [Object], [Shape])
    Net.__init__(self, "Query", **kwargs)


class QueryShape(Function, Net):
  def __init__(self, name="QueryShape", **kwargs):
    Function.__init__(self, [Object], [Shape])
    Net.__init__(self, "QueryShape", **kwargs)


class QuerySize(Function, Net):
  def __init__(self, name="QuerySize", **kwargs):
    Function.__init__(self, [Object], [Size])
    Net.__init__(self, "QuerySize", **kwargs)


class QueryMaterial(Function, Net):
  def __init__(self, name="QueryMaterial", **kwargs):
    Function.__init__(self, [Object], [Material])
    Net.__init__(self, "QueryMaterial", **kwargs)


class QueryColor(Function, Net):
  def __init__(self, name="QueryColor", **kwargs):
    Function.__init__(self, [Object], [Color])
    Net.__init__(self, "QueryColor", **kwargs)


class Same(Function, Net):
  def __init__(self, name="Same", **kwargs):
    Function.__init__(self, [ObjectSet, Object], [ObjectSet])
    Net.__init__(self, "Same", **kwargs)


class SameShape(Function, Net):
  def __init__(self, name="SameShape", **kwargs):
    Function.__init__(self, [ObjectSet, Object], [ObjectSet])
    Net.__init__(self, "SameShape", **kwargs)


class SameSize(Function, Net):
  def __init__(self, name="SameSize", **kwargs):
    Function.__init__(self, [ObjectSet, Object], [ObjectSet])
    Net.__init__(self, "SameSize", **kwargs)


class SameMaterial(Function, Net):
  def __init__(self, name="SameMaterial", **kwargs):
    Function.__init__(self, [ObjectSet, Object], [ObjectSet])
    Net.__init__(self, "SameMaterial", **kwargs)


class SameColor(Function, Net):
  def __init__(self, name="SameColor", **kwargs):
    Function.__init__(self, [ObjectSet, Object], [ObjectSet])
    Net.__init__(self, "SameColor", **kwargs)
