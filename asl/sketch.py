"Sketching"
import torch
import torch.nn as nn

from asl.type import Function
from asl.modules.modules import expand_consts


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
    self.add_module("interface", model)

  def ref_losses(self):
    "Return map from an interface to a loss saying whether its correct"

  def choice_losses(self):
    "For any hard choices"

  def losses(self):
    "Union of reflosses and choice_losses"

  def forward(self, *xs):
    return self.sketch(*xs, **self.model)
