import asl
from aslbench.clevr.data import parse

def eval_string(func_string, inputs):
  f = eval(func_string)
  return f(*inputs)


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
      value_inputs = [value_transform(parse(val)) for val in call['value_inputs']]
      all_inputs = inputs + value_inputs
      if fname in ["same_shape", "same_color", "same_material", "same_size"]:
        all_inputs = [scene_object_set] + all_inputs
      if fname == "relate":
        all_inputs = [relations] + all_inputs

      fouts[i] = apply(fname, all_inputs)

  return fouts[-1]
