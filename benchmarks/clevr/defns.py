from asl.type import Function, Type
import torch.nn as nn
from asl.archs.mlp import MLPNet
from asl.net import Net
import benchmarks.clevr.clevr as clevr
from typing import Union

ObjectSetLatent = clevr.TensorClevrObjectSet
ObjectLatent = clevr.TensorClevrObject
RelationsLatent = clevr.TensorRelations
RelationLatent = clevr.RelationOneHot1D

ColorLatent = clevr.ColorOneHot1D
MaterialLatent = clevr.MaterialOneHot1D
ShapeLatent = clevr.ShapeOneHot1D
SizeLatent = clevr.SizeOneHot1D
PropertyLatent = Union[ColorLatent, MaterialLatent, ShapeLatent, SizeLatent]

BooleanLatent = clevr.BooleanOneHot1D
IntegerLatent = clevr.IntegerOneHot1D

class Unique(Function, Net):
  def __init__(self="Unique", name="Unique", **kwargs):
    Function.__init__(self, [ObjectSetLatent], [ObjectLatent])
    Net.__init__(self, " name", **kwargs)


class Relate(Function, Net):
  def __init__(self="Relate", name="Relate", **kwargs):
    Function.__init__(self, [RelationsLatent, ObjectLatent, RelationLatent],
                            [ObjectSetLatent])
    Net.__init__(self, " name", **kwargs)


class Count(Function, Net):
  def __init__(self, name="Count", **kwargs):
    Function.__init__(self, [ObjectSetLatent], [IntegerLatent])
    Net.__init__(self, "Count", **kwargs)


class Exist(Function, Net):
  def __init__(self, name="Exist", **kwargs):
    Function.__init__(self, [ObjectSetLatent], [BooleanLatent])
    Net.__init__(self, "Exist", **kwargs)


class Filter(Function, Net):
  def __init__(self, name="Filter", **kwargs):
    Function.__init__(self, [ObjectSetLatent, PropertyLatent], [ObjectSetLatent])
    Net.__init__(self, "Filter", **kwargs)


class FilterSize(Function, Net):
  def __init__(self, name="FilterSize", **kwargs):
    Function.__init__(self, [ObjectSetLatent, SizeLatent], [ObjectSetLatent])
    Net.__init__(self, "FilterSize", **kwargs)


class FilterColor(Function, Net):
  def __init__(self, name="FilterColor", **kwargs):
    Function.__init__(self, [ObjectSetLatent, ColorLatent], [ObjectSetLatent])
    Net.__init__(self, "FilterColor", **kwargs)


class FilterMaterial(Function, Net):
  def __init__(self, name="FilterMaterial", **kwargs):
    Function.__init__(self, [ObjectSetLatent, MaterialLatent], [ObjectSetLatent])
    Net.__init__(self, "FilterMaterial", **kwargs)


class FilterShape(Function, Net):
  def __init__(self, name="FilterShape", **kwargs):
    Function.__init__(self, [ObjectSetLatent, ShapeLatent], [ObjectSetLatent])
    Net.__init__(self, "FilterShape", **kwargs)


class Intersect(Function, Net):
  def __init__(self, name="Intersect", **kwargs):
    Function.__init__(self, [ObjectSetLatent, ObjectSetLatent], [ObjectSetLatent])
    Net.__init__(self, "Intersect", **kwargs)


class Union(Function, Net):
  def __init__(self, name="Union", **kwargs):
    Function.__init__(self, [ObjectSetLatent, ObjectSetLatent], [ObjectSetLatent])
    Net.__init__(self, "Union", **kwargs)


class GreaterThan(Function, Net):
  def __init__(self, name="GreaterThan", **kwargs):
    Function.__init__(self, [IntegerLatent, IntegerLatent], [BooleanLatent])
    Net.__init__(self, "GreaterThan", **kwargs)


class LessThan(Function, Net):
  def __init__(self, name="LessThan", **kwargs):
    Function.__init__(self, [IntegerLatent, IntegerLatent], [BooleanLatent])
    Net.__init__(self, "LessThan", **kwargs)


class EqualInteger(Function, Net):
  def __init__(self, name="EqualInteger", **kwargs):
    Function.__init__(self, [IntegerLatent, IntegerLatent], [BooleanLatent])
    Net.__init__(self, "EqualInteger", **kwargs)


class Equal(Function, Net):
  def __init__(self, name="Equal", **kwargs):
    Function.__init__(self, [WHAT, WHAT], [BooleanLatent])
    Net.__init__(self, "Equal", **kwargs)


class EqualMaterial(Function, Net):
  def __init__(self, name="EqualMaterial", **kwargs):
    Function.__init__(self, [MaterialLatent, MaterialLatent], [BooleanLatent])
    Net.__init__(self, "EqualMaterial", **kwargs)


class EqualSize(Function, Net):
  def __init__(self, name="EqualSize", **kwargs):
    Function.__init__(self, [SizeLatent, SizeLatent], [BooleanLatent])
    Net.__init__(self, "EqualSize", **kwargs)


class EqualShape(Function, Net):
  def __init__(self, name="EqualShape", **kwargs):
    Function.__init__(self, [ShapeLatent, ShapeLatent], [BooleanLatent])
    Net.__init__(self, "EqualShape", **kwargs)


class EqualColor(Function, Net):
  def __init__(self, name="EqualColor", **kwargs):
    Function.__init__(self, [ColorLatent, ColorLatent], [BooleanLatent])
    Net.__init__(self, "EqualColor", **kwargs)


class QueryShape(Function, Net):
  def __init__(self, name="QueryShape", **kwargs):
    Function.__init__(self, [ObjectLatent], [ShapeLatent])
    Net.__init__(self, "QueryShape", **kwargs)


class QuerySize(Function, Net):
  def __init__(self, name="QuerySize", **kwargs):
    Function.__init__(self, [ObjectLatent], [SizeLatent])
    Net.__init__(self, "QuerySize", **kwargs)


class QueryMaterial(Function, Net):
  def __init__(self, name="QueryMaterial", **kwargs):
    Function.__init__(self, [ObjectLatent], [MaterialLatent])
    Net.__init__(self, "QueryMaterial", **kwargs)


class QueryColor(Function, Net):
  def __init__(self, name="QueryColor", **kwargs):
    Function.__init__(self, [ObjectLatent], [ColorLatent])
    Net.__init__(self, "QueryColor", **kwargs)


class SameShape(Function, Net):
  def __init__(self, name="SameShape", **kwargs):
    Function.__init__(self, [ObjectSetLatent, ObjectLatent], [ObjectSetLatent])
    Net.__init__(self, "SameShape", **kwargs)


class SameSize(Function, Net):
  def __init__(self, name="SameSize", **kwargs):
    Function.__init__(self, [ObjectSetLatent, ObjectLatent], [ObjectSetLatent])
    Net.__init__(self, "SameSize", **kwargs)


class SameMaterial(Function, Net):
  def __init__(self, name="SameMaterial", **kwargs):
    Function.__init__(self, [ObjectSetLatent, ObjectLatent], [ObjectSetLatent])
    Net.__init__(self, "SameMaterial", **kwargs)


class SameColor(Function, Net):
  def __init__(self, name="SameColor", **kwargs):
    Function.__init__(self, [ObjectSetLatent, ObjectLatent], [ObjectSetLatent])
    Net.__init__(self, "SameColor", **kwargs)
