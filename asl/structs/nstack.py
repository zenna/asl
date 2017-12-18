"Stack Data Structure trained from a reference implementation"
from asl.type import Function
from asl.modules.modules import ConstantNet, ModuleDict
from asl.archs.convnet import ConvNet
from asl.util.misc import cuda
from torch import nn


class Push(Function):
  "Push Function for Stack"

  def __init__(self, stack_type, item_type):
    super(Push, self).__init__([stack_type, item_type], [stack_type])


class Pop(Function):
  "Pop Function for Stack"

  def __init__(self, stack_type, item_type):
    super(Pop, self).__init__([stack_type], [stack_type, item_type])


def type_check(xs, types):
  assert len(xs) == len(types)
  for i, x in enumerate(xs):
    same_size = xs[i].size()[1:] == types[i].size
    assert same_size
  return xs

def list_push(stack, element):
  stack = stack.copy()
  stack.append(element)
  return (stack, )


def list_pop(stack):
  stack = stack.copy()
  item = stack.pop()
  return (stack, item)

list_empty = []


def neural_stack(element_type, stack_type):
  push_img = PushNet(stack_type, element_type)
  pop_img = PopNet(stack_type, element_type)
  empty_stack = ConstantNet(stack_type)
  neural_ref = ModuleDict({"push": push_img,
                           "pop": pop_img,
                           "empty": empty_stack})
  cuda(neural_ref)
  return neural_ref


def ref_stack():
  return {"push": list_push, "pop": list_pop, "empty": []}
