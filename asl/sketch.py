"Sketching"
from enum import Enum
import torch
import torch.nn as nn
import asl
from asl.loss import vec_dist
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


class Mode(Enum):
  NOMODE = 0
  REF = 1
  NEURAL = 2


class Sketch(Function, nn.Module):
  "Sketch Composition of Modules"

  def __init__(self, in_types, out_types, model, ref_model):
    super(Sketch, self).__init__(in_types, out_types)
    self.mods = []
    self.ref_observes = []
    self.observes = []
    self.model = model
    self.ref_model = ref_model
    self.add_module("interface", model)
    self.mode = Mode.NOMODE

  def observe(self, value, label=''):
    if self.mode is Mode.NOMODE:
      print("cant observe values without choosing mode")
      raise ValueError
    elif self.mode is Mode.NEURAL:
      self.observes.append(value)
    else:
      self.ref_observes.append(value)
    return value

  def observe_loss(self):
    "Loss between observed and references"
    nobs = len(self.ref_observes)
    if nobs != len(self.observes):
      raise ValueError
    if nobs == 0:
      print("No observes found")
      raise ValueError

    return vec_dist(self.observes, self.ref_observes)

  def clear_observes(self):
    self.ref_observes = []
    self.observes = []

  def forward(self, *xs):
    self.mode = Mode.NEURAL
    res = self.sketch(*xs, **self.model)
    self.mode = Mode.NOMODE
    return res

  def forward_ref(self, *xs):
    self.mode = Mode.REF
    res = self.sketch(*xs, **self.ref_model)
    self.mode = Mode.NOMODE
    return res


def inner_loss_gen(sketch, items_iter, ref_items_iter):
  sketch.forward(items_iter)
  sketch.forward_ref(ref_items_iter)
  loss = sketch.observe_loss()
  asl.log.log("observes", sketch.observes)
  asl.log.log("ref_observes", sketch.ref_observes)
  sketch.clear_observes()
  return loss


def loss_gen_gen(sketch, tl, itr_transform=None):
  "Computes the reference loss"
  if itr_transform is None:
    itr = iter
  else:
    itr = lambda loader: asl.util.misc.imap(itr_transform, iter(loader))
  items_iter = itr(tl)
  ref_items_iter = itr(tl)

  def loss_gen():
    nonlocal items_iter, ref_items_iter
    try:
      return inner_loss_gen(sketch, items_iter, ref_items_iter)
    except StopIteration:
      print("End of Epoch, restarting iterators")
      sketch.clear_observes()
      items_iter = itr(tl)
      ref_items_iter = itr(tl)
      return inner_loss_gen(sketch, items_iter, ref_items_iter)

  return loss_gen
