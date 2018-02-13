"Stack learned from reference"
import torch
import random
import os
from typing import List
import asl
from asl.modules.modules import ConstantNet, ModuleDict
from asl.structs.nstack import list_push, list_pop, list_empty, Push, Pop
from torch import optim, nn
import common
from stdargs import optim_sampler, arch_sampler
from mnist import mnist_size, Mnist, dist, refresh_mnist
from omniglot import omniglot_size, OmniGlot, dist, refresh_omniglot
from asl.callbacks import every_n
from multipledispatch import dispatch
from asl.loss import mean

## Training
def train_stack(opt):
  if opt["dataset"] == "omniglot":
    stack_size = asl.util.repl(mnist_size, 0, opt["nchannels"])
    ItemType = OmniGlot
    dataloader = asl.util.omniglotloader
    refresh_data = lambda dl: refresh_omniglot(dl, nocuda=opt["nocuda"])
  else:
    stack_size = asl.util.repl(mnist_size, 0, opt["nchannels"])
    ItemType = Mnist
    dataloader = asl.util.mnistloader
    refresh_data = lambda dl: refresh_mnist(dl, nocuda=opt["nocuda"])

  ## Data structures and functions
  class MatrixStack(asl.Type):
    typesize = stack_size

  tracegen = opt["tracegen"]
  trace = tracegen(opt["nitems"], opt["nrounds"])
  push = Push(MatrixStack, ItemType, arch=opt["arch"], arch_opt=opt["arch_opt"])
  pop = Pop(MatrixStack, ItemType, arch=opt["arch"], arch_opt=opt["arch_opt"])
  empty = ConstantNet(MatrixStack,
                      requires_grad=opt["learn_constants"],
                      init=opt["init"])

  def ref_sketch(items, r, runstate):
    return trace(items, r, runstate, push=list_push, pop=list_pop, empty=list_empty)

  class StackSketch(asl.Sketch):
    def sketch(self, items, r, runstate):
      """Example stack trace"""
      return trace(items, r, runstate, push=push, pop=pop, empty=empty)

  stack_sketch = StackSketch([List[OmniGlot]], [OmniGlot])
  nstack = ModuleDict({"push": push,
                       "pop": pop,
                       "empty": empty,
                       "stack_sketch": stack_sketch})

  # Cuda that shit
  asl.cuda(nstack, opt["nocuda"])

  # Hack to add random object as input to traces
  def refresh_with_random(x):
    # import pdb; pdb.set_trace()
    refreshed = refresh_data(x)
    return [refreshed[0], random.Random(0)]

  # Loss
  it = dataloader(opt["batch_size"], normalize=opt["normalize"])
  loss_gen = asl.single_ref_loss(stack_sketch,
                                 ref_sketch,
                                 it,
                                 refresh_with_random,
                                 accum=opt["accum"])
  if opt["learn_constants"]:
    parameters = nstack.parameters()
  else:
    parameters = torch.nn.ModuleList([push, pop]).parameters()

  return common.trainmodel(opt, nstack, loss_gen, parameters)