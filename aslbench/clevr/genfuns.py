import random
from typing import Union
import asl
from . import clevr

def func_types():
  ObjectSetLatent = clevr.TensorClevrObjectSet
  ObjectLatent = clevr.TensorClevrObject
  RelationsLatent = clevr.TensorRelations
  RelationLatent = clevr.RelationOneHot1D

  encoding = random.choice([asl.OneHot1D, asl.OneHot2D])
  ColorLatent = asl.encode(clevr.Color, encoding, (10, 10))
  MaterialLatent = asl.encode(clevr.Material, encoding, (10, 10))
  ShapeLatent = asl.encode(clevr.Shape, encoding, (10, 10))
  SizeLatent = asl.encode(clevr.Size, encoding, (10, 10))
  PropertyLatent = Union[ColorLatent, MaterialLatent, ShapeLatent, SizeLatent]

  BooleanLatent = asl.encode(clevr.Boolean, encoding, (10, 10))
  IntegerLatent = asl.encode(clevr.Integer, encoding, (10, 10))
  return funcs(ObjectSetLatent,
               ObjectLatent,
               RelationsLatent,
               RelationLatent,
               ColorLatent,
               MaterialLatent,
               ShapeLatent,
               SizeLatent,
               PropertyLatent,
               BooleanLatent,
               IntegerLatent)


def funcs(ObjectSetLatent,
          ObjectLatent,
          RelationsLatent,
          RelationLatent,
          ColorLatent,
          MaterialLatent,
          ShapeLatent,
          SizeLatent,
          PropertyLatent,
          BooleanLatent,
          IntegerLatent):

  class Unique(asl.Function, asl.Net):
    def __init__(self="Unique", name="Unique", **kwargs):
      asl.Function.__init__(self, [ObjectSetLatent], [ObjectLatent])
      asl.Net.__init__(self, " name", **kwargs)


  class Relate(asl.Function, asl.Net):
    def __init__(self="Relate", name="Relate", **kwargs):
      asl.Function.__init__(self, [RelationsLatent, ObjectLatent, RelationLatent],
                                  [ObjectSetLatent])
      asl.Net.__init__(self, " name", **kwargs)


  class Count(asl.Function, asl.Net):
    def __init__(self, name="Count", **kwargs):
      asl.Function.__init__(self, [ObjectSetLatent], [IntegerLatent])
      asl.Net.__init__(self, "Count", **kwargs)


  class Exist(asl.Function, asl.Net):
    def __init__(self, name="Exist", **kwargs):
      asl.Function.__init__(self, [ObjectSetLatent], [BooleanLatent])
      asl.Net.__init__(self, "Exist", **kwargs)


  class Filter(asl.Function, asl.Net):
    def __init__(self, name="Filter", **kwargs):
      asl.Function.__init__(self, [ObjectSetLatent, PropertyLatent], [ObjectSetLatent])
      asl.Net.__init__(self, "Filter", **kwargs)


  class FilterSize(asl.Function, asl.Net):
    def __init__(self, name="FilterSize", **kwargs):
      asl.Function.__init__(self, [ObjectSetLatent, SizeLatent], [ObjectSetLatent])
      asl.Net.__init__(self, "FilterSize", **kwargs)


  class FilterColor(asl.Function, asl.Net):
    def __init__(self, name="FilterColor", **kwargs):
      asl.Function.__init__(self, [ObjectSetLatent, ColorLatent], [ObjectSetLatent])
      asl.Net.__init__(self, "FilterColor", **kwargs)


  class FilterMaterial(asl.Function, asl.Net):
    def __init__(self, name="FilterMaterial", **kwargs):
      asl.Function.__init__(self, [ObjectSetLatent, MaterialLatent], [ObjectSetLatent])
      asl.Net.__init__(self, "FilterMaterial", **kwargs)


  class FilterShape(asl.Function, asl.Net):
    def __init__(self, name="FilterShape", **kwargs):
      asl.Function.__init__(self, [ObjectSetLatent, ShapeLatent], [ObjectSetLatent])
      asl.Net.__init__(self, "FilterShape", **kwargs)


  class Intersect(asl.Function, asl.Net):
    def __init__(self, name="Intersect", **kwargs):
      asl.Function.__init__(self, [ObjectSetLatent, ObjectSetLatent], [ObjectSetLatent])
      asl.Net.__init__(self, "Intersect", **kwargs)


  class Union(asl.Function, asl.Net):
    def __init__(self, name="Union", **kwargs):
      asl.Function.__init__(self, [ObjectSetLatent, ObjectSetLatent], [ObjectSetLatent])
      asl.Net.__init__(self, "Union", **kwargs)


  class GreaterThan(asl.Function, asl.Net):
    def __init__(self, name="GreaterThan", **kwargs):
      asl.Function.__init__(self, [IntegerLatent, IntegerLatent], [BooleanLatent])
      asl.Net.__init__(self, "GreaterThan", **kwargs)


  class LessThan(asl.Function, asl.Net):
    def __init__(self, name="LessThan", **kwargs):
      asl.Function.__init__(self, [IntegerLatent, IntegerLatent], [BooleanLatent])
      asl.Net.__init__(self, "LessThan", **kwargs)


  class EqualInteger(asl.Function, asl.Net):
    def __init__(self, name="EqualInteger", **kwargs):
      asl.Function.__init__(self, [IntegerLatent, IntegerLatent], [BooleanLatent])
      asl.Net.__init__(self, "EqualInteger", **kwargs)


  class Equal(asl.Function, asl.Net):
    def __init__(self, name="Equal", **kwargs):
      asl.Function.__init__(self, [PropertyLatent, PropertyLatent], [BooleanLatent])
      asl.Net.__init__(self, "Equal", **kwargs)


  class EqualMaterial(asl.Function, asl.Net):
    def __init__(self, name="EqualMaterial", **kwargs):
      asl.Function.__init__(self, [MaterialLatent, MaterialLatent], [BooleanLatent])
      asl.Net.__init__(self, "EqualMaterial", **kwargs)


  class EqualSize(asl.Function, asl.Net):
    def __init__(self, name="EqualSize", **kwargs):
      asl.Function.__init__(self, [SizeLatent, SizeLatent], [BooleanLatent])
      asl.Net.__init__(self, "EqualSize", **kwargs)


  class EqualShape(asl.Function, asl.Net):
    def __init__(self, name="EqualShape", **kwargs):
      asl.Function.__init__(self, [ShapeLatent, ShapeLatent], [BooleanLatent])
      asl.Net.__init__(self, "EqualShape", **kwargs)


  class EqualColor(asl.Function, asl.Net):
    def __init__(self, name="EqualColor", **kwargs):
      asl.Function.__init__(self, [ColorLatent, ColorLatent], [BooleanLatent])
      asl.Net.__init__(self, "EqualColor", **kwargs)


  class QueryShape(asl.Function, asl.Net):
    def __init__(self, name="QueryShape", **kwargs):
      asl.Function.__init__(self, [ObjectLatent], [ShapeLatent])
      asl.Net.__init__(self, "QueryShape", **kwargs)


  class QuerySize(asl.Function, asl.Net):
    def __init__(self, name="QuerySize", **kwargs):
      asl.Function.__init__(self, [ObjectLatent], [SizeLatent])
      asl.Net.__init__(self, "QuerySize", **kwargs)


  class QueryMaterial(asl.Function, asl.Net):
    def __init__(self, name="QueryMaterial", **kwargs):
      asl.Function.__init__(self, [ObjectLatent], [MaterialLatent])
      asl.Net.__init__(self, "QueryMaterial", **kwargs)


  class QueryColor(asl.Function, asl.Net):
    def __init__(self, name="QueryColor", **kwargs):
      asl.Function.__init__(self, [ObjectLatent], [ColorLatent])
      asl.Net.__init__(self, "QueryColor", **kwargs)


  class SameShape(asl.Function, asl.Net):
    def __init__(self, name="SameShape", **kwargs):
      asl.Function.__init__(self, [ObjectSetLatent, ObjectLatent], [ObjectSetLatent])
      asl.Net.__init__(self, "SameShape", **kwargs)


  class SameSize(asl.Function, asl.Net):
    def __init__(self, name="SameSize", **kwargs):
      asl.Function.__init__(self, [ObjectSetLatent, ObjectLatent], [ObjectSetLatent])
      asl.Net.__init__(self, "SameSize", **kwargs)


  class SameMaterial(asl.Function, asl.Net):
    def __init__(self, name="SameMaterial", **kwargs):
      asl.Function.__init__(self, [ObjectSetLatent, ObjectLatent], [ObjectSetLatent])
      asl.Net.__init__(self, "SameMaterial", **kwargs)


  class SameColor(asl.Function, asl.Net):
    def __init__(self, name="SameColor", **kwargs):
      asl.Function.__init__(self, [ObjectSetLatent, ObjectLatent], [ObjectSetLatent])
      asl.Net.__init__(self, "SameColor", **kwargs)

  return {'Unique': Unique,
          'Relate': Relate,
          'Count': Count,
          'Exist': Exist,
          'Filter': Filter,
          'FilterSize': FilterSize,
          'FilterColor': FilterColor,
          'FilterMaterial': FilterMaterial,
          'FilterShape': FilterShape,
          'Intersect': Intersect,
          'Union': Union,
          'GreaterThan': GreaterThan,
          'LessThan': LessThan,
          'EqualInteger': EqualInteger,
          'Equal': Equal,
          'EqualMaterial': EqualMaterial,
          'EqualSize': EqualSize,
          'EqualShape': EqualShape,
          'EqualColor': EqualColor,
          'QueryShape': QueryShape,
          'QuerySize': QuerySize,
          'QueryMaterial': QueryMaterial,
          'QueryColor': QueryColor,
          'SameShape': SameShape,
          'SameSize': SameSize,
          'SameMaterial': SameMaterial,
          'SameColor': SameColor}
