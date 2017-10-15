"Stack Data Structure trained from a reference implementation"
from collections import deque
from asl.type import Function
from asl.modules.templates import VarConvNet
from asl.modules.modules import ConstantNet, ModuleDict
from asl.util.misc import cuda
from torch import nn


class Enqueue(Function):
  "Enqueue Function for Stack"

  def __init__(self, queue_type, item_type):
    super(Enqueue, self).__init__([queue_type, item_type], [queue_type])


class Dequeue(Function):
  "Dequeue Function for Stack"

  def __init__(self, queue_type, item_type):
    super(Dequeue, self).__init__([queue_type], [queue_type, item_type])


# class NetFunction(nn.Module):
#
#   def __init__(self, name, template=VarConvNet, module=None):
#     if module is None:
#       self.module = template(self.in_sizes(), self.out_sizes())
#     else:
#       self.module = module
#     self.add_module(name, self.module)
#
#   def forward(self, *xs):
#     return self.module(*xs)
#
#
# class EnqueueNet(Enqueue, NetFunction):
#   def __init__(self, queue_type, item_type, template, module=None):
#     Enqueue.__init__(self, queue_type, item_type)
#     NetFunction.__init__(self, "Enqeue", template, module)
#
#
# class DequeueNet(Dequeue, NetFunction):
#   def __init__(self, queue_type, item_type, template, module=None):
#     Dequeue.__init__(self, queue_type, item_type)
#     NetFunction.__init__(self, "Deqeue", template, module)


class EnqueueNet(Enqueue, nn.Module):
  def __init__(self, stack_type, item_type, module=None, template=VarConvNet,
               template_opt=None):
    super(EnqueueNet, self).__init__(stack_type, item_type)
    if module is None:
      template_opt = {} if template_opt is None else template_opt
      self.module = template(self.in_sizes(), self.out_sizes(), **template_opt)
    else:
      self.module = module
    self.add_module("Enqueue", self.module)

  def forward(self, x, y):
    return self.module.forward(x, y)


class DequeueNet(Dequeue, nn.Module):
  def __init__(self, stack_type, item_type, module=None, template=VarConvNet,
               template_opt=None):
    super(DequeueNet, self).__init__(stack_type, item_type)
    if module is None:
      template_opt = {} if template_opt is None else template_opt
      self.module = template(self.in_sizes(), self.out_sizes(), **template_opt)
    else:
      self.module = module
    self.add_module("Dequeue", self.module)


  def forward(self, x):
    return self.module.forward(x)



def list_enqueue(queue, element):
  queue = queue.copy()
  queue.append(element)
  return (queue, )


def list_dequeue(queue):
  queue = queue.copy()
  item = queue.popleft()
  return (queue, item)


def neural_queue(element_type, queue_type):
  enqueue_img = EnqueueNet(queue_type, element_type)
  dequeue_img = DequeueNet(queue_type, element_type)
  empty_queue = ConstantNet(queue_type)
  neural_ref = ModuleDict({"enqueue": enqueue_img,
                           "dequeue": dequeue_img,
                           "empty": empty_queue})
  cuda(neural_ref)
  return neural_ref


def ref_queue():
  return {"enqueue": list_enqueue, "dequeue": list_dequeue, "empty": deque()}
