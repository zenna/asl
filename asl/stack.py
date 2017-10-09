"Stack Data Structure trained from a reference implementation"
import itertools
from type import Type, Function, FunctionType, Constant

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from train import train

from util import draw, trainloader

class Push(Function):
  "Push Function for Stack"

  def __init__(self, stack_type, item_type):
    super(Push, self).__init__()
    self.stack_type = stack_type
    self.item_type = item_type

  def type(self):
    "Function Type"
    return FunctionType([self.stack_type, self.item_type], [self.stack_type])


class Pop(Function):
  "Pop Function for Stack"

  def __init__(self, stack_type, item_type):
    super(Pop, self).__init__()
    self.stack_type = stack_type
    self.item_type = item_type

  def type(self):
    "Function Type"
    return FunctionType([self.stack_type], [self.item_type])


class PushNet(Push, nn.Module):
  def __init__(self, stack_type, item_type, stack_channels=1, img_channels=1):
    super(PushNet, self).__init__(stack_type, item_type)
    in_channels = stack_channels + img_channels
    out_channels = stack_channels
    nf = 16
    self.conv1 = nn.Conv2d(in_channels, nf, 3, padding=1)
    self.convmid = nn.Conv2d(nf, nf, 3, padding=1)
    self.conv2 = nn.Conv2d(nf, stack_channels, 3, padding=1)

  def forward(self, x, y):
    x = torch.cat([x, y], dim=1)
    # import pdb; pdb.set_trace()
    x = F.elu(self.conv1(x))
    x = F.elu(self.conv2(x))
    return (x,)


class PopNet(Pop, nn.Module):
  def __init__(self, stack_type, item_type, stack_channels=1, img_channels=1):
    super(PopNet, self).__init__(stack_type, item_type)
    self.stack_channels = stack_channels
    self.img_channels = img_channels
    out_channels = stack_channels + img_channels
    nf = 16
    self.conv1 = nn.Conv2d(stack_channels, nf, 3, padding=1)
    self.convmid = nn.Conv2d(nf, nf, 3, padding=1)
    self.conv2 = nn.Conv2d(nf, out_channels, 3, padding=1)

  def forward(self, *x):
    channel_dim = 1
    x, = x
    x = F.elu(self.conv1(x))
    x = F.elu(self.conv2(x))
    (img, stack) = x.split(self.img_channels, channel_dim)
    return (stack, img)


def list_push(stack, element):
  stack = stack.copy()
  stack.append(element)
  return (stack, )


def list_pop(stack):
  stack = stack.copy()
  item = stack.pop()
  return (stack, item)


def empty():
  return []


def stack_trace(items, push, pop, empty):
    """Example stack trace"""
    items = [Variable(data[0].cuda()) for data in list(itertools.islice(items, 3))]
    # if expand_empty:
    empty = empty()
    observes = []
    stack = empty
    # print("mean", items[0].mean())
    (stack,) = push(stack, items[0])
    (stack,) = push(stack, items[1])
    # (stack,) = push(stack, items[2])
    (pop_stack, pop_item) = pop(stack)
    observes.append(pop_item)
    (pop_stack, pop_item) = pop(pop_stack)
    observes.append(pop_item)
    return observes


def plot_empty(i, data):
  if i % 50 == 0:
    draw(data["empty"].value)


def main():
  matrix_stack = Type("Stack", (28, 28), dtype="float32")
  mnist_type = Type("mnist_type", (28, 28), dtype="float32")
  push_img = PushNet(matrix_stack, mnist_type)
  pop_img = PopNet(matrix_stack, mnist_type)
  push_img.cuda()
  pop_img.cuda()
  empty_stack = Constant(matrix_stack, Variable(torch.rand(1, 1, 28, 28).cuda(), requires_grad=True))
  stack_ref = {"push": list_push, "pop": list_pop, "empty": empty}
  neural_ref = {"push": push_img, "pop": pop_img, "empty": empty_stack}

  batch_size = 32
  train(stack_trace, trainloader(batch_size), stack_ref, neural_ref, batch_size,
        callbacks=[plot_empty])


if __name__ == "__main__":
  main()
