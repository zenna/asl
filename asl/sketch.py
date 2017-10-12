"Sketching"
from copy import copy
from typing import List
import torch
import torch.nn as nn
from torch.autograd import Variable

from asl.type import Function
from asl.util import iterget, cuda

from asl.type import Type
from asl.structs.nstack import neural_stack, ref_stack
from asl.util import draw, trainloader, as_img
from asl.train import trainloss, log_append
from asl.modules import expand_consts
from asl.callbacks import every_n

from torch import optim

def soft_ch(choices, which):
  "Soft choice of elements of choices"
  nchoices = len(choices)
  if nchoices == 1:
    return choices[0]
  choices = expand_consts(choices)
  decision_vec = which(nchoices)
  decision_vec = torch.nn.Softmax()(decision_vec)
  scaled = [choices[i] * decision_vec[0, i] for i in range(nchoices)]
  return sum(scaled)


class Sketch(Function, nn.Module):
  "Sketch Composition of Modules"

  def __init__(self, in_types, out_types, model, ref_model):
    super(Sketch, self).__init__(in_types, out_types)
    self.mods = []
    self.refobserves = []
    self.neurobserves = []
    self.model = model
    self.ref_mdoel = ref_model
    for name, module in self.model.items():
      self.add_module(name, module)

  def ref_losses(self):
    "Return map from an interface to a loss saying whether its correct"

  def choice_losses(self):
    "For any hard choices"

  def losses(self):
    "Union of reflosses and choice_losses"

  def forward(self, *xs):
    return self.sketch(*xs, **self.model)


class ReverseSketch(Sketch):
  "Sketch for reversal of list of elements"

  def __init__(self, element_type, model, ref_model):
    super(ReverseSketch, self).__init__([List[element_type]],
                                        [List[element_type]],
                                        model,
                                        ref_model)
    self.choice1 = Variable(cuda(torch.rand(1, 2)), requires_grad=True)

  def choice1f(self, outlen):
    proj = Variable(cuda(torch.rand(2, outlen)))
    return torch.matmul(self.choice1, proj)

  def sketch(self, items, push, pop, empty):
    stack_chs = [empty]
    out_items = []
    item_chs = copy(items)
    for _ in range(3):
      (stack,) = push(soft_ch(stack_chs, self.choice1f),
                      soft_ch(item_chs, self.choice1f))
      stack_chs.append(stack)

    for _ in range(3):
      (stack, item) = pop(soft_ch(stack_chs, self.choice1f))
      stack_chs.append(stack)
      out_items.append(item)

    return out_items



def test_reverse_sketch():
  "Test Reverse Sketch"
  batch_size = 128
  tl = trainloader(batch_size)
  items_iter = iter(tl)

  matrix_stack = Type("Stack", (1, 28, 28), dtype="float32")
  mnist_type = Type("mnist_type", (1, 28, 28), dtype="float32")
  nstack = neural_stack(mnist_type, matrix_stack)
  refstack = ref_stack(mnist_type, matrix_stack)
  rev_sketch = ReverseSketch(Type, nstack, refstack)

  rev_items_iter = iter(tl)
  optimizer = optim.Adam(rev_sketch.parameters(), lr=0.0001)

  def plot_items(i, log, writer, **kwargs):
    writer.add_image('fwd/1', log['forward'][0][0][0], i)
    writer.add_image('fwd/2', log['forward'][0][1][0], i)
    writer.add_image('fwd/3', log['forward'][0][2][0], i)
    writer.add_image('rev/1', log['reverse'][0][0][0], i)
    writer.add_image('rev/2', log['reverse'][0][1][0], i)
    writer.add_image('rev/3', log['reverse'][0][2][0], i)
    writer.add_image('out/1', log['out'][0][0][0], i)
    writer.add_image('out/2', log['out'][0][1][0], i)
    writer.add_image('out/3', log['out'][0][2][0], i)


  def loss_gen():
    nonlocal items_iter, rev_items_iter
    # Refresh hte iterators if they run out
    try:
      items = iterget(items_iter, 3)
      rev_items = iterget(rev_items_iter, 3)
    except StopIteration:
      print("End of Epoch")
      items_iter = iter(tl)
      rev_items_iter = iter(tl)
      items = iterget(items_iter, 3)
      rev_items = iterget(rev_items_iter, 3)

    out_items = rev_sketch(items)
    rev_items.reverse()
    log_append("forward", items)
    log_append("reverse", rev_items)
    log_append("out", out_items)

    losses = [nn.MSELoss()(out_items[i], rev_items[i]) for i in range(3)]
    loss = sum(losses)
    return loss

  trainloss(loss_gen, optimizer, [every_n(plot_items, 100)])


if __name__ == "__main__":
  test_reverse_sketch()
