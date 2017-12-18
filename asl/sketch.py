"Sketching"
import torch
import torch.nn as nn
from asl.type import Function
from asl.modules.modules import expand_consts

def soft_ch(choices, decision_vec):
  "Soft choice of elements of choices"
  nchoices = len(choices)
  if nchoices == 1:
    return choices[0]
  choices = expand_consts(choices)
  decision_vec = torch.nn.Softmax()(decision_vec)
  scaled = [choices[i] * decision_vec[0, i] for i in range(nchoices)]
  return sum(scaled)


def soft_ch_var(choices, which):
  "Soft choice of elements of choices"
  decision_vec = which(len(choices))
  return soft_ch(choices, decision_vec)


class Sketch(Function, nn.Module):
  "Sketch Composition of Modules"

  def __init__(self, in_types, out_types):
    super(Sketch, self).__init__(in_types, out_types)
    nn.Module.__init__(self)

  def forward(self, *xs):
    res = self.sketch(*xs)
    return res
