"Stack Data Structure trained from a reference implementation"
import asl
from asl.type import Function
from asl.modules.modules import ConstantNet, ModuleDict
from asl.archs.convnet import ConvNet
from asl.util.misc import cuda

class Push(asl.Function, asl.Net):
  def __init__(self,  StackType, ItemType, name="Push", **kwargs):
    asl.Function.__init__(self, [StackType, ItemType], [StackType])
    asl.Net.__init__(self, name, **kwargs)

class Pop(asl.Function, asl.Net):
  def __init__(self, StackType, ItemType, name="Pop", **kwargs):
    asl.Function.__init__(self, [StackType], [StackType, ItemType])
    asl.Net.__init__(self, name, **kwargs)

def list_push(stack, element):
  stack = stack.copy()
  stack.append(element)
  return (stack, )


def list_pop(stack):
  stack = stack.copy()
  item = stack.pop()
  return (stack, item)

list_empty = []