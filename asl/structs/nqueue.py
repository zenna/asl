"Queue Reference Implementation"
import asl
from collections import deque
from asl.type import Function

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

empty_queue = deque()