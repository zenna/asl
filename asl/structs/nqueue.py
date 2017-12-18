"Stack Data Structure trained from a reference implementation"
import asl
from collections import deque
from asl.type import Function
from asl.archs.convnet import ConvNet
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
