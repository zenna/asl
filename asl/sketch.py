"Sketching"
import torch
import torch.nn as nn
from torch.autograd import Variable

from asl.type import Function

# Tricky?
# - How to do hard choice?
# - If we do something like my experiment, we split based on the batch
# - Vecttor of inputs or steram or what?
# - Training Net

def soft_ch(choices, which):
  "Soft choice of elements of choices"
  if len(choices) == 1:
    return choices[0]
  else:
    decision_vec = which(len(choices))
    decision_vec = torch.nn.Softmax()(decision_vec)
    res
    n = len(choices)
    reshape_size = [batch_size, n] + [1 for i in range(n)]
    dec_tensor = decision_vec.view(*reshape_size)
    for i in range(len(choices)):
      res = res * decision_vec[:, i]

    return res

def test():
  batch_size = 8
  a = torch.rand(batch_size, 1, 2, 3)
  b = torch.rand(batch_size, 1, 2, 3)
  c = torch.rand(batch_size, 1, 2, 3)
  xs = [a, b, c]
  n = len(xs)
  decision_vec = torch.rand(batch_size, n)
  reshape_size = [batch_size, n] + [1 for i in range(n)]
  q = decision_vec.view(*reshape_size)
  q2 = q[:, 0, 0:1, 0:1, 0:1]

def hard_ch(type, choices, which):
  "Hard choice"


class Sketch(Function, nn.Module):
  "Sketch Composition of Modules"

  def __init__(self, in_types, out_types, sketch):
    super(Sketch).__init__(in_types, out_types)
    self.mods = []
    self.refobserves = []
    self.neurobserves = []

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

  def __init__(self):
    super(ReverseSketch).__init__(in_types, out_types)
    self.choice1 = Variable(torch.rand(2), requires_grad=True)

  def choice1f(self, outlen):
    proj = Variable(torch.rand(2, outlen))
    return torch.matmul(self.choice1, proj)

  def sketch(self, items, push, pop, empty):
    stack_chs = [empty]
    out_items = []
    item_chs = []
    for _ in range(3):
      (stack,) = push(soft_ch(stack_chs, self.choice1f),
                      soft_ch([item_chs], self.choice1f))
      stack_chs.append(stack)

    for _ in range(3):
      (stack, item) = pop(soft_ch(stack_chs, self.choice1f))
      stack_chs.append(stack)


    return out_items

def test_reverse_sketch():
  "Test Reverse Sketch"
