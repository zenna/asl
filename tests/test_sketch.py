from typing import List
from copy import copy
from asl.structs.nstack import neural_stack, ref_stack
from asl.type import Type
from asl.sketch import Sketch, soft_ch
from asl.util.misc import trainloader, cuda, iterget
from asl.log import log_append
from asl.train import train
from asl.callbacks import every_n

from torch import optim, nn
import torch
from torch.autograd import Variable


class ReverseSketch(Sketch):
  "Sketch for reversal of list of elements"

  def __init__(self, element_type, model, ref_model):
    super(ReverseSketch, self).__init__([List[element_type]],
                                        [List[element_type]],
                                        model,
                                        ref_model)
    self.choice1 = nn.Parameter(torch.rand(1, 2), requires_grad=True)

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
  cuda(rev_sketch)
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

  train(loss_gen, optimizer, [every_n(plot_items, 100)])


if __name__ == "__main__":
  test_reverse_sketch()
