"Stack Data Structure trained from a reference implementation"
from asl.type import Function
from asl.modules import VarConvNet, ConstantNet, ModuleDict
from asl.util import cuda

from torch import nn


class Push(Function):
  "Push Function for Stack"

  def __init__(self, stack_type, item_type):
    super(Push, self).__init__([stack_type, item_type], [stack_type])

class Pop(Function):
  "Pop Function for Stack"

  def __init__(self, stack_type, item_type):
    super(Pop, self).__init__([stack_type], [stack_type, item_type])


class PushNet(Push, nn.Module):
  def __init__(self, stack_type, item_type, stack_channels=1, img_channels=1):
    super(PushNet, self).__init__(stack_type, item_type)
    self.module = VarConvNet(self.in_sizes(), self.out_sizes())

  def forward(self, x, y):
    return self.module.forward(x, y)


class PopNet(Pop, nn.Module):
  def __init__(self, stack_type, item_type, stack_channels=1, img_channels=1):
    super(PopNet, self).__init__(stack_type, item_type)
    self.module = VarConvNet(self.in_sizes(), self.out_sizes())

  def forward(self, x):
    return self.module.forward(x)


def list_push(stack, element):
  stack = stack.copy()
  stack.append(element)
  return (stack, )


def list_pop(stack):
  stack = stack.copy()
  item = stack.pop()
  return (stack, item)


def neural_stack(element_type, stack_type):
  push_img = PushNet(stack_type, element_type)
  pop_img = PopNet(stack_type, element_type)
  empty_stack = ConstantNet(stack_type)
  neural_ref = ModuleDict({"push": push_img,
                           "pop": pop_img,
                           "empty": empty_stack})
  cuda(neural_ref)
  return neural_ref


def ref_stack(element_type, stack_type):
  return {"push": list_push, "pop": list_pop, "empty": []}
