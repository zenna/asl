"Stack Data Structure trained from a reference implementation"
import itertools

from asl.type import Type, Function
from asl.train import trainloss, log, log_append
from asl.modules import VarConvNet, ConstantNet, ModuleDict
from asl.callbacks import tb_loss, every_n
from asl.util import draw, trainloader, as_img, iterget

import torch
from torch.autograd import Variable
from torch import optim, nn


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


def stack_trace(items, push, pop, empty):
    """Example stack trace"""
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


def neural_stack(element_type, stack_type):
  push_img = PushNet(stack_type, element_type)
  pop_img = PopNet(stack_type, element_type)
  push_img.cuda()
  pop_img.cuda()
  empty_stack = ConstantNet(stack_type)
  neural_ref = ModuleDict({"push": push_img,
                           "pop": pop_img,
                           "empty": empty_stack})
  return neural_ref


def ref_stack(element_type, stack_type):
  return {"push": list_push, "pop": list_pop, "empty": []}


def observe_loss(criterion, obs, refobs, state=None):
  "MSE between observations from reference and training stack"
  total_loss = 0.0
  losses = [criterion(obs[i], refobs[i]) for i in range(len(obs))]
  print([loss[0].data[0] for loss in losses])
  total_loss = sum(losses)
  return total_loss


def main():
  batch_size = 128
  tl = trainloader(batch_size)
  dataiter = iter(tl)
  refiter = iter(tl)
  matrix_stack = Type("Stack", (1, 28, 28), dtype="float32")
  mnist_type = Type("mnist_type", (1, 28, 28), dtype="float32")
  nstack = neural_stack(mnist_type, matrix_stack)
  refstack = ref_stack(mnist_type, matrix_stack)

  criterion = nn.MSELoss()
  optimizer = optim.Adam(nstack.parameters(), lr=0.0001)

  def loss_gen():
    try:
      items = iterget(dataiter, 3)
      ref_items = iterget(refiter, 3)
    except StopIteration:
      print("End of Epoch")
      items_iter = iter(tl)
      ref_items_iter = iter(tl)
      items = iterget(items_iter, 3)
      ref_items = iterget(ref_items_iter, 3)


    observes = stack_trace(items, **nstack)
    refobserves = stack_trace(ref_items, **refstack)
    return observe_loss(criterion, observes, refobserves)

  trainloss(loss_gen, optimizer,)

if __name__ == "__main__":
  main()
