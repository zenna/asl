"Stack Data Structure trained from a reference implementation"
import itertools

from asl.type import Type, Function, Constant
from asl.train import train, log, log_append
from asl.nets import VarConvNet
from asl.callbacks import tb_loss, every_n
from asl.util import draw, trainloader, as_img

import torch
from torch.autograd import Variable
import torch.nn as nn


class Push(Function):
  "Push Function for Stack"

  def __init__(self, stack_type, item_type):
    super(Push, self).__init__([stack_type, item_type], [stack_type])
    self.stack_type = stack_type
    self.item_type = item_type


class Pop(Function):
  "Pop Function for Stack"

  def __init__(self, stack_type, item_type):
    super(Pop, self).__init__([stack_type], [stack_type, item_type])
    self.stack_type = stack_type
    self.item_type = item_type


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


def list_empty():
  return []


def stack_trace(items, push, pop, empty):
    """Example stack trace"""
    items = [Variable(data[0].cuda()) for data in list(itertools.islice(items, 3))]
    # if expand_empty:
    empty = empty()
    log_append("empty", empty)

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
    log_append("observes", observes)
    return observes


def plot_observes(i, log, writer, **kwargs):
  "Show the empty set in tensorboardX"
  refobserves = log['observes'][0]
  nobserves = log['observes'][1]
  for j in range(len(refobserves)):
    writer.add_image('compare{}/ref'.format(j), refobserves[j][0], i)
    writer.add_image('compare{}/neural'.format(j), nobserves[j][0], i)


def plot_empty(i, log, writer, **kwargs):
  "Show the empty set in tensorboardX"
  img = log['empty'][0][0]
  writer.add_image('EmptySet', img, i)


def main():
  batch_size = 128
  matrix_stack = Type("Stack", (1, 28, 28), dtype="float32")
  mnist_type = Type("mnist_type", (1, 28, 28), dtype="float32")
  push_img = PushNet(matrix_stack, mnist_type)
  pop_img = PopNet(matrix_stack, mnist_type)
  push_img.cuda()
  pop_img.cuda()
  empty_stack = Constant(matrix_stack,
                        Variable(torch.rand(1, 1, 28, 28).cuda(),
                                 requires_grad=True),
                         batch_size)
  stack_ref = {"push": list_push, "pop": list_pop, "empty": list_empty}
  neural_ref = {"push": push_img, "pop": pop_img, "empty": empty_stack}

  tl = trainloader(batch_size)
  train(stack_trace, tl, stack_ref, neural_ref, batch_size,
        callbacks=[tb_loss,
                   every_n(plot_empty, 100),
                   every_n(plot_observes, 100)],
        nepochs=500)


if __name__ == "__main__":
  main()
