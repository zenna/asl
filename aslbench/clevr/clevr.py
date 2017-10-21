"Clevr Benchmark"
import ijson
import os
from enum import Enum
import asl
import asl.util as util
from asl.util import maybe_expand
from asl.encoding import OneHot1D, Encoding, OneHot2D, onehot1d, onehot2d
from torch.autograd import Variable
import torch

# Shape
class Shape():
  pass


class ShapeOneHot1D(Shape, OneHot1D):
  typesize = (8,)
  def __init__(self, value, expand_one=True):
    self.value = maybe_expand(ShapeOneHot1D, value, expand_one)


class ShapeEnum(Shape, Enum):
  cube = 0
  sphere = 1
  cylinder = 2


# Material
class Material():
  pass


class MaterialOneHot1D(Material, OneHot1D):
  typesize = (8,)
  def __init__(self, value, expand_one=True):
    self.value = maybe_expand(MaterialOneHot1D, value, expand_one)


class MaterialOneHot2D(Shape, OneHot2D):
  typesize = (8,)
  def __init__(self, value, expand_one=True):
    self.value = maybe_expand(MaterialOneHot2D, value, expand_one)


class MaterialEnum(Material, Enum):
  metal = 0
  rubber = 1

# Size
class Size():
  pass


class SizeOneHot1D(Size, OneHot1D):
  typesize = (8,)
  def __init__(self, value, expand_one=True):
    self.value = maybe_expand(SizeOneHot1D, value, expand_one)



class SizeOneHot2D(Shape, OneHot2D):
  typesize = (8, 8)
  def __init__(self, value, expand_one=True):
    self.value = maybe_expand(SizeOneHot2D, value, expand_one)


class SizeEnum(Size, Enum):
  small = 0
  large = 1


class Color():
  pass


class ColorOneHot1D(Color, OneHot1D):
  typesize = (8,)
  def __init__(self, value, expand_one=True):
    self.value = maybe_expand(ColorOneHot1D, value, expand_one)


class ColorEnum(Color, Enum):
  red = 0
  green = 1
  gray = 2
  yellow = 3
  blue = 4
  cyan = 5
  brown = 6
  purple = 7


class Relation():
  pass


class RelationOneHot1D(Relation, OneHot1D):
  typesize = (8, )
  def __init__(self, value, expand_one=True):
    self.value = maybe_expand(RelationOneHot1D, value, expand_one)


class RelationEnum(Relation, Enum):
  left = 0
  right = 1
  front = 2
  behind = 3


class Boolean():
  pass


class BooleanOneHot1D(Boolean, OneHot1D):
  typesize = (2,)
  def __init__(self, value, expand_one=True):
    self.value = maybe_expand(BooleanOneHot1D, value, expand_one)


class BooleanOneHot2D(Boolean, OneHot2D):
  typesize = (2, 2)
  def __init__(self, value, expand_one=True):
    self.value = maybe_expand(BooleanOneHot1D, value, expand_one)


class BooleanEnum(Boolean, Enum):
  "Boolean"
  yes = 0
  no = 1


class Integer():
  pass


class IntegerOneHot1D(Integer, OneHot1D):
  typesize = (11,)
  def __init__(self, value, expand_one=True):
    self.value = maybe_expand(IntegerOneHot1D, value, expand_one)


class ClevrObject():
  def __init__(self, color, material, shape, size):
    self.color = color
    self.material = material
    self.shape = shape
    self.size = size

  def from_json(json):
    return ClevrObject(color=ColorEnum[json['color']],
                       material=MaterialEnum[json['material']],
                       shape=ShapeEnum[json['shape']],
                       size=SizeEnum[json['size']])

  def tensor(self):
    return Variable(util.cuda(util.onehotmany([self.color.value,
                                                    self.size.value,
                                                    self.material.value,
                                                    self.shape.value], 8)))

class TensorClevrObject(Encoding):
  "An embedding of an object as a tensor"
  typesize = (4, 8)

  def __init__(self, value, expand_one=True):
    self.value = maybe_expand(TensorClevrObject, value, expand_one)


class ClevrObjectSet():
  def __init__(self, objects):
    assert isinstance(objects, list)
    assert len(objects) == 0 or isinstance(objects[0], ClevrObject)
    self.objects = objects

  def from_json(objects):
    return ClevrObjectSet(list(map(ClevrObject.from_json, objects)))

  def tensor(self, max_n_objects=10):
    obj_tensors = [t.tensor().expand(1, 4, 8) for t in self.objects]
    ndummies = max_n_objects - len(obj_tensors)
    assert ndummies >= 0
    dummies = [Variable(util.cuda(torch.zeros(1, 4, 8))) for i in range(ndummies)]
    return TensorClevrObjectSet(torch.cat(obj_tensors + dummies, 0))


class TensorClevrObjectSet(Encoding):
  "An embedding of an object as a tensor"
  typesize = (10, 4, 8)

  def __init__(self, value, expand_one=True):
    self.value = maybe_expand(TensorClevrObjectSet, value, expand_one)


class Relations():
  "Python implementation of a relation"
  def __init__(self, relations, listform):
    self.relations = relations
    self.listform = listform

  def from_json(json, object_set):
    relations = {}
    for (i, obj) in enumerate(object_set.objects):
      hello = {}
      for rel in RelationEnum:
        objsids = json['relationships'][rel.name][i]
        hello[rel] = [object_set.objects[j] for j in objsids]
      relations[obj] = hello

    return Relations(relations, json['relationships'])

  def tensor(self):
    nrels = 4
    maxnobjs = 10
    rel_ten = torch.zeros(nrels, maxnobjs, maxnobjs)
    for (i, rel) in enumerate(['behind', 'front', 'left', 'right']):
      for (j, obj1rels) in enumerate(self.listform[rel]):
        for obj2 in obj1rels:
          rel_ten[i, j, obj2] = 1.0

    return TensorRelations(Variable(util.cuda(rel_ten)))


class TensorRelations(Encoding):
  "Relations represented as a matrix"
  typesize = (4, 10, 10)

  def __init__(self, value, expand_one=True):
    self.value = maybe_expand(TensorRelations, value, expand_one)


class SceneGraph():
  "Python Implementation of a scene graph"
  def __init__(self, object_set, relations):
    self.object_set = object_set
    self.relations = relations

  def from_json(json):
    "construct a scene graph from the jason"
    object_set = ClevrObjectSet.from_json(json['objects'])
    relations = Relations.from_json(json, object_set)
    return SceneGraph(object_set, relations)

def scene(scene_graph):
  return scene_graph.object_set

def unique(object_set):
  if len(object_set.objects) != 1:
    raise ValueError
  else:
    return object_set.objects[0]

def relate(relations, object, relation):
  return ClevrObjectSet(relations.relations[object][relation])

def count(object_set):
  return len(object_set.objects)

def exist(object_set):
  return len(object_set.objects) > 0

# Filter functions
def filter_size(object_set, size):
  return ClevrObjectSet(list(filter(lambda obj: obj.size == size,
                                    object_set.objects)))


def filter_color(object_set, color):
  return ClevrObjectSet(list(filter(lambda obj: obj.color == color,
                                    object_set.objects)))


def filter_material(object_set, material):
  return ClevrObjectSet(list(filter(lambda obj: obj.material == material,
                                    object_set.objects)))


def filter_shape(object_set, shape):
  return ClevrObjectSet(list(filter(lambda obj: obj.shape == shape,
                                    object_set.objects)))


def list_intersect(a, b):
  return list(set(a).intersection(set(b)))


def list_union(a, b):
  return list(set(a).union(set(b)))


def intersect(object_set1, object_set2):
  return ClevrObjectSet(list_intersect(object_set1.objects, object_set2.objects))


def union(object_set1, object_set2):
  return ClevrObjectSet(list_union(object_set1.objects, object_set2.objects))


def greater_than(a, b):
  return a > b


def less_than(a, b):
  return a < b


def equal_integer(a, b):
  return a == b


def equal_material(a, b):
  return a == b


def equal_size(a, b):
  return a == b


def equal_shape(a, b):
  return a == b


def equal_color(a, b):
  return a == b


def query_shape(object):
  return object.shape


def query_size(object):
  return object.size


def query_material(object):
  return object.material


def query_color(object):
  return object.color


def rem(object_set, object):
  return ClevrObjectSet([obj for obj in object_set.objects if obj != object])


def same_shape(scene_object_set, object):
  return rem(filter_shape(scene_object_set, query_shape(object)), object)


def same_size(scene_object_set, object):
  return rem(filter_size(scene_object_set, query_size(object)), object)


def same_material(scene_object_set, object):
  return rem(filter_material(scene_object_set, query_material(object)), object)


def same_color(scene_object_set, object):
  return rem(filter_color(scene_object_set, query_color(object)), object)


def eval_string(func_string, inputs):
  f = eval(func_string)
  return f(*inputs)

VALUE = {}
VALUE.update({x.name: x for x in ColorEnum})
VALUE.update({x.name: x for x in MaterialEnum})
VALUE.update({x.name: x for x in ShapeEnum})
VALUE.update({x.name: x for x in SizeEnum})
VALUE.update({x.name: x for x in RelationEnum})
VALUE.update({x.name: x for x in BooleanEnum})

def interpret(json,
              scene_object_set,
              relations,
              apply=eval_string,
              value_transform=asl.util.misc.identity):
  "interpret the json function spec"
  fouts = [() for i in json]
  for i, call in enumerate(json):
    fname = call['function']
    if fname == "scene":
      fouts[i] = scene_object_set
    else:
      inputs = [fouts[i] for i in call['inputs']]
      value_inputs = [value_transform(VALUE[val]) for val in call['value_inputs']]
      all_inputs = inputs + value_inputs
      if fname in ["same_shape", "same_color", "same_material", "same_size"]:
        all_inputs = [scene_object_set] + all_inputs
      if fname == "relate":
        all_inputs = [relations] + all_inputs

      fouts[i] = apply(fname, all_inputs)

  return fouts[-1]


num_to_string = {'0': 0,
                 '1': 1,
                 '2': 2,
                 '3': 3,
                 '4': 4,
                 '5': 5,
                 '6': 6,
                 '7': 7,
                 '8': 8,
                 '9': 9,
                 '10':10}


def ans_tensor(ans):
  "Convert the query answer into a tensor"
  if ans in num_to_string:
    value = num_to_string[ans]
    return IntegerOneHot1D(Variable(util.cuda(util.onehot(value, 11, 1))))
  else:
    value = VALUE[ans]
    {ColorEnum: }
    # need to pass a mpapign from the python data types to their encoding
    # PARMATERIZE THIS CHOICE
    return onehot2d(value)
  return


def proghasfunc(func, program):
 return any(list(map(lambda call: call['function'] == func, program)))


ref_clevr = {'unique': unique,
             'relate': relate,
             'count': count,
             'exist': exist,
             'filter_size': filter_size,
             'filter_color': filter_color,
             'filter_material': filter_material,
             'filter_shape': filter_shape,
             'list_intersect': list_intersect,
             'list_union': list_union,
             'intersect': intersect,
             'union': union,
             'greater_than': greater_than,
             'less_than': less_than,
             'equal_integer': equal_integer,
             'equal_material': equal_material,
             'equal_size': equal_size,
             'equal_shape': equal_shape,
             'equal_color': equal_color,
             'query_shape': query_shape,
             'query_size': query_size,
             'query_material': query_material,
             'query_color': query_color,
             'same_shape': same_shape,
             'same_size': same_size,
             'same_material': same_material,
             'same_color': same_color}
