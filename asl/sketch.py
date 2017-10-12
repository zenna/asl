"Sketching"
from copy import copy
from typing import List
import torch
import torch.nn as nn
from torch.autograd import Variable

from asl.type import Function, expand_consts
from asl.util import iterget, cuda

from asl.type import Type
from asl.structs.nstack import neural_stack, ref_stack
# Tricky?
# - How to do hard choice?
# - If we do something like my experiment, we split based on the batch
# - Vecttor of inputs or steram or what?
# - Training Net

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


def hard_ch(type, choices, which):
  "Hard choice"


class Sketch(Function, nn.Module):
  "Sketch Composition of Modules"

  def __init__(self, in_types, out_types, model, ref_model):
    super(Sketch, self).__init__(in_types, out_types)
    self.mods = []
    self.refobserves = []
    self.neurobserves = []
    self.model = model
    self.ref_mdoel = ref_model

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
    import pdb; pdb.set_trace()
    for _ in range(3):
      (stack,) = push(soft_ch(stack_chs, self.choice1f),
                      soft_ch(item_chs, self.choice1f))
      stack_chs.append(stack)

    for _ in range(3):
      (stack, item) = pop(soft_ch(stack_chs, self.choice1f))
      stack_chs.append(stack)

    return out_items


from asl.util import draw, trainloader, as_img

def test_reverse_sketch():
  "Test Reverse Sketch"
  batch_size = 128
  tl = trainloader(batch_size)
  tliter = iter(tl)
  items = iterget(tliter, 3)
  matrix_stack = Type("Stack", (1, 28, 28), dtype="float32")
  mnist_type = Type("mnist_type", (1, 28, 28), dtype="float32")
  nstack = neural_stack(mnist_type, matrix_stack)
  refstack = ref_stack(mnist_type, matrix_stack)
  rev_sketch = ReverseSketch(Type, nstack, refstack)
  rev_sketch(items)

if __name__ == "__main__":
  test_reverse_sketch()
