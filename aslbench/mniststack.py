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

# Comment out
# OmniGlot = Mnist
# omniglot_size = mnist_size
# refresh_omniglot = refresh_mnist
# dataloader = asl.util.mnistloader
dataloader = asl.util.omniglotloader

def tracegen(nitems, nrounds):
  print("Making stack trace with {} items and {} rounds".format(nitems, nrounds))
  def trace(items, runstate, push, pop, empty):
    """Example stack trace"""
    # import pdb; pdb.set_trace()
    asl.log_append("empty", empty)
    stack = empty
    for nr in range(nrounds):
      for i in range(nitems):
        (stack,) = push(stack, next(items))
        # print("BLIP!")
        asl.log_append("{}/internal".format(runstate['mode']), stack)

      for j in range(nitems):
        (stack, pop_item) = pop(stack)
        asl.observe(pop_item, "pop.{}.{}".format(nr, j), runstate)
        # print("BLIP!")
        asl.log_append("{}/internal".format(runstate['mode']), stack)
      
    return pop_item
  
  return trace

## Data structures and functions
class MatrixStack(asl.Type):
  typesize = omniglot_size

## Training
def train_stack(opt):
  trace = tracegen(opt["nitems"], opt["nrounds"])
  push = Push(MatrixStack, OmniGlot, arch=opt["arch"], arch_opt=opt["arch_opt"])
  pop = Pop(MatrixStack, OmniGlot, arch=opt["arch"], arch_opt=opt["arch_opt"])
  empty = ConstantNet(MatrixStack,
                      requires_grad=opt["learn_constants"],
                      init=opt["init"])

  def ref_sketch(items, runstate):
    return trace(items, runstate, push=list_push, pop=list_pop, empty=list_empty)

  class StackSketch(asl.Sketch):
    def sketch(self, items, runstate):
      """Example stack trace"""
      return trace(items, runstate, push=push, pop=pop, empty=empty)

  stack_sketch = StackSketch([List[OmniGlot]], [OmniGlot])
  nstack = ModuleDict({"push": push,
                       "pop": pop,
                       "empty": empty,
                       "stack_sketch": stack_sketch})

  # Cuda that shit
  asl.cuda(nstack, opt["nocuda"])

  # Loss
  omniglotiter = dataloader(opt["batch_size"])
  loss_gen = asl.single_ref_loss(stack_sketch,
                                 ref_sketch,
                                 omniglotiter,
                                 refresh_omniglot,
                                 accum=opt["accum"])
  if opt["learn_constants"]:
    parameters = nstack.parameters()
  else:
    parameters = torch.nn.ModuleList([push, pop]).parameters()

  return common.trainmodel(opt, nstack, loss_gen, parameters)

## Samplers

def stack_args(parser):
  parser.add_argument('--nitems', type=int, default=3, metavar='NI',
                      help='number of iteems in trace (default: 3)')
  parser.add_argument('--nrounds', type=int, default=1, metavar='NR',
                      help='number of rounds in trace')

def stack_optspace():
  return {"nrounds": [1, 2],
          # "nitems": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20],
          # "batch_size": [32, 64, 128, 256, 512],
          "nitems": [3],
          "batch_size": [16],
          "learn_constants": [True],
          "accum": [mean],
          "init": [#torch.nn.init.uniform,
                   torch.nn.init.normal,
                  #  torch.nn.init.xavier_normal,
                  #  torch.nn.init.xavier_uniform,
                  #  torch.nn.init.kaiming_normal,
                  #  torch.nn.init.kaiming_uniform,
                   torch.ones_like,
                  #  torch.zeros_like
                   ],
          "arch_opt": arch_sampler,
          "optim_args": optim_sampler}

def runoptsgen(nsamples):
  # Delaying computation of this value because we dont know nsamples yet
  return asl.prodsample(stack_optspace(),
                        to_enum=["nitems"],
                        to_sample=["init", "nrounds", "batch_size", "lr", "accum", "learn_constants"],
                        to_sample_merge=["arch_opt", "optim_args"],
                        nsamples=nsamples)

if __name__ == "__main__":
  thisfile = os.path.abspath(__file__)
  res = common.trainloadsave(thisfile, train_stack, runoptsgen, stack_args)