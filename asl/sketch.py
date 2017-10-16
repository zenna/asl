"Sketching"
import torch
import torch.nn as nn

import asl
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


def mean(xs):
  return sum(xs) / len(xs)


def dist(x, y):
  return nn.MSELoss()(x, y) # TODO: Specialize this by type


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
    self.mode = None

  def observe(self, value):
    if self.mode is None:
      print("cant observe values without choosing mode")
      raise ValueError
    elif self.mode is True:
      self.observes.append(value)
    else:
      self.ref_observes.append(value)

  def observe_loss(self, accumulate=mean):
    "Loss between observed and references"
    nobs = len(self.ref_observes)
    if nobs != len(self.observes):
      raise ValueError
    if nobs == 0:
      print("No observes found")
      raise ValueError

    losses = [dist(self.observes[i], self.ref_observes[i]) for i in range(nobs)]
    return accumulate(losses)

  def clear_observes(self):
    self.ref_observes = []
    self.observes = []

  def forward(self, *xs):
    self.mode = True
    res = self.sketch(*xs, **self.model)
    self.mode = None
    return res

  def forward_ref(self, *xs):
    self.mode = False
    res = self.sketch(*xs, **self.ref_model)
    self.mode = None
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
