import ijson
import os
from enum import Enum
import asl
from asl.util.io import datadir
import asl.util.torch
from asl.util.misc import take
from asl.util.misc import cuda
import torch
from torch.autograd import Variable


def clevr_iter(clevr_root,
               data_type,
               train=True):
  path = os.path.join(clevr_root, data_type)
  train_val = "train" if train else "val"
  if train:
    path = os.path.join(path, "CLEVR_{}_{}.json".format(train_val, data_type))
  else:
    path = os.path.join(path, "CLEVR_{}_{}.json".format(train_val, data_type))

  f = open(path, "r")
  return ijson.items(f, "{}.item".format(data_type))


def questions_iter(clevr_root=os.path.join(datadir(), "CLEVR_v1.0"),
                   train=True):
  "Iterator over question dataset"
  return clevr_iter(clevr_root, "questions", train)


def scenes_iter(clevr_root=os.path.join(datadir(), "CLEVR_v1.0"),
                train=True):
  "Iterator over scenes"
  return clevr_iter(clevr_root, "scenes", train)


# def data_iter(batch_size, train=True):
#   "Iterates paired scene and question data"
#   if batch_size % 10 != 0:
#     raise ValueError
#
#   qitr = questions_iter(train=train)
#   sitr = scenes_iter(train=train)
#   ndraws = batch_size // 10
#
#   while True:
#     rel_tens = []
#     obj_set_tens = []
#     progs = []
#     answers = []
#     for i in range(ndraws):
#       s1 = next(sitr)
#       scene1 = SceneGraph.from_json(s1)
#       rel_ten = scene1.relations.tensor()
#       objset_ten = scene1.object_set.tensor()
#       rel_tens.append(rel_ten)
#       obj_set_tens.append(objset_ten)
#       ques = take(qitr, 10)
#       ten_progs = [q['program'] for q in ques]
#       ten_answers = [q['answer'] for q in ques]
#       progs.append(ten_progs)
#       answers.append(ten_answers)
#
#     yield progs, obj_set_tens, rel_tens, answers


def data_iter(batch_size, train=True):
  "Iterates paired scene and question data"
  qitr = questions_iter(train=train)
  sitr = scenes_iter(train=train)

  while True:
    rel_tens = []
    obj_set_tens = []
    progs = []
    answers = []
    for i in range(batch_size):
      qi = next(qitr)
      si = next(sitr)
      scenei = SceneGraph.from_json(si)
      rel_ten = scenei.relations.tensor()
      rel_tens.append(rel_ten.expand(1, *rel_ten.size()))
      obj_ten = scenei.object_set.tensor()
      obj_set_tens.append(cuda(obj_ten.expand(1, *obj_ten.size())))
      progs.append(qi['program'])
      answers.append(qi['answer'])

    yield progs, obj_set_tens, rel_tens, answers

class Shape(Enum):
  def tensor(self):
    return Variable(cuda(asl.util.torch.onehot(self.value, 8, 1)))
  cube = 0
  sphere = 1
  cylinder = 2


class Material(Enum):
  def tensor(self):
    return Variable(cuda(asl.util.torch.onehot(self.value, 8, 1)))
  metal = 0
  rubber = 1


class Size(Enum):
  def tensor(self):
    return Variable(cuda(asl.util.torch.onehot(self.value, 8, 1)))
  small = 0
  large = 1


class Color(Enum):
  def tensor(self):
    return Variable(cuda(asl.util.torch.onehot(self.value, 8, 1)))
  red = 0
  green = 1
  gray = 2
  yellow = 3
  blue = 4
  cyan = 5
  brown = 6
  purple = 7


class Relation(Enum):
  def tensor(self):
    return Variable(cuda(asl.util.torch.onehot(self.value, 8, 1)))
  left = 0
  right = 1
  front = 2
  behind = 3


class Boolean(Enum):
  def tensor(self):
    return Variable(cuda(asl.util.torch.onehot(self.value, 2, 1)))
  "Boolean"
  yes = 0
  no = 1


class Integer(Enum):
  def tensor(self):
    return Variable(cuda(asl.util.torch.onehot(self.value, 1, 1)))
  "Boolean"
  yes = 0
  no = 1



class ClevrObject():
  def __init__(self, color, material, shape, size):
    self.color = color
    self.material = material
    self.shape = shape
    self.size = size

  def from_json(json):
    return ClevrObject(color=Color[json['color']],
                       material=Material[json['material']],
                       shape=Shape[json['shape']],
                       size=Size[json['size']])

  def tensor(self):
    return Variable(cuda(asl.util.torch.onehotmany([self.color.value,
                                                    self.size.value,
                                                    self.material.value,
                                                    self.shape.value], 8)))


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
    dummies = [Variable(cuda(torch.zeros(1, 4, 8))) for i in range(ndummies)]
    return torch.cat(obj_tensors + dummies, 0)


class Relations():
  "Python implementation of a relation"
  def __init__(self, relations, listform):
    self.relations = relations
    self.listform = listform

  def from_json(json, object_set):
    relations = {}
    for (i, obj) in enumerate(object_set.objects):
      hello = {}
      for rel in Relation:
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

    return Variable(cuda(rel_ten))


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
VALUE.update({x.name: x for x in Color})
VALUE.update({x.name: x for x in Material})
VALUE.update({x.name: x for x in Shape})
VALUE.update({x.name: x for x in Size})
VALUE.update({x.name: x for x in Relation})
VALUE.update({x.name: x for x in Boolean})


# VALUE.update({'left': 'left',
#               'right': 'right',
#               'front': 'front',
#               'behind': 'behind'})
#

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
                 '9': 9}


def ans_tensor(ans):
  "Convert the query answer into a tensor"
  if ans in num_to_string:
    value = num_to_string[ans]
    return Variable(cuda(asl.util.torch.onehot(value, 10, 1)))
  else:
    value = VALUE[ans]
    return value.tensor()
  return


def test_interpret():
  qitr = questions_iter()
  sitr = scenes_iter()

  while True:
    s1 = next(sitr)
    # 10 questions per scene it seems
    for i in range(10):
      q1 = next(qitr)
      scene = SceneGraph.from_json(s1)
      res = interpret(q1['program'], scene.object_set, scene.relations)
      print(res, q1['answer'])

# test_interpret()

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
#
# qitr = questions_iter()
# sitr = scenes_iter()
# s1 = next(sitr)
# sc1 = SceneGraph.from_json(s1)
# os = sc1.object_set
# os.tensor()
# o1 = sc1.object_set.objects[0]
# o1.tensor()
# # o1.torch






# • Convert json into python program
# • Make types for all interface functiosn
# • continuous representation of scene
# • scene graph into program that builds one
# • validate pipeline on dataset using exact python implementations



# 1. Convert json into scene graph
# eitehr continuiize scene graph.
  # onehot object properties
  # one hot relation ship properties, into some big tensor

# write program that constructs scene graph using primitives
# so now i have some embedding of a scene graph

# next is to take a function program
# parse that into a python progra
# interpret it with neural counter part to get output
#

# How to convert pytho
