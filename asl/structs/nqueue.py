"Stack Data Structure trained from a reference implementation"
from asl.type import Function
from asl.modules import VarConvNet, ConstantNet, ModuleDict
from asl.util.misc import cuda

from torch import nn
from collections import deque

class Enqueue(Function):
  "Enqueue Function for Stack"

  def __init__(self, queue_type, item_type):
    super(Enqueue, self).__init__([queue_type, item_type], [queue_type])

class Dequeue(Function):
  "Dequeue Function for Stack"

  def __init__(self, queue_type, item_type):
    super(Dequeue, self).__init__([queue_type], [queue_type, item_type])


class EnqueueNet(Enqueue, nn.Module):
  def __init__(self, queue_type, item_type1):
    super(EnqueueNet, self).__init__(queue_type, item_type)
    self.module = VarConvNet(self.in_sizes(), self.out_sizes())

  def forward(self, x, y):
    return self.module.forward(x, y)


class DequeueNet(Dequeue, nn.Module):
  def __init__(self, queue_type, item_type):
    super(DequeueNet, self).__init__(queue_type, item_type)
    self.module = VarConvNet(self.in_sizes(), self.out_sizes())

  def forward(self, x):
    return self.module.forward(x)


def list_enqueue(queue, element):
  queue = queue.copy()
  queue.append(element)
  return (queue, )


def list_dequeue(queue):
  queue = queue.copy()
  item = queue.dequeueleft()
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


def ref_queue(element_type, queue_type):
  return {"enqueue": list_enqueue, "dequeue": list_dequeue, "empty": deque()}
