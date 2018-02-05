"Stack learned from reference"
import torch
import random
import os
from typing import List
import asl
from asl.modules.modules import ConstantNet, ModuleDict
from asl.structs.nstack import list_push, list_pop, list_empty
from torch import optim, nn
import common
from stdargs import optim_sampler, arch_sampler
from mnist import mnist_size, Mnist, dist, refresh_mnist
from asl.callbacks import every_n
from multipledispatch import dispatch
from asl.loss import mean

def tracegen(nitems, nrounds):
  print("Making stack trace with {} items and {} rounds".format(nitems, nrounds))
  def trace(items, runstate, push, pop, empty):
    """Example stack trace"""
    asl.log_append("empty", empty)
    stack = empty
    for nr in range(nrounds):
      for i in range(nitems):
        (stack,) = push(stack, next(items))
        asl.log_append("{}/internal".format(runstate['mode']), stack)

      for j in range(nitems):
        (stack, pop_item) = pop(stack)
        asl.observe(pop_item, "pop.{}.{}".format(nr, j), runstate)
        asl.log_append("{}/internal".format(runstate['mode']), stack)
      
    return pop_item
  
  return trace

## Data structures and functions
class MatrixStack(asl.Type):
  typesize = mnist_size

class Push(asl.Function, asl.Net):
  def __init__(self, name="Push", **kwargs):
    asl.Function.__init__(self, [MatrixStack, Mnist], [MatrixStack])
    asl.Net.__init__(self, name, **kwargs)

class Pop(asl.Function, asl.Net):
  def __init__(self, name="Pop", **kwargs):
    asl.Function.__init__(self, [MatrixStack], [MatrixStack, Mnist])
    asl.Net.__init__(self, name, **kwargs)

## Training
def train_stack(opt):
  trace = tracegen(opt["nitems"], opt["nrounds"])
  push = Push(arch=opt["arch"], arch_opt=opt["arch_opt"])
  pop = Pop(arch=opt["arch"], arch_opt=opt["arch_opt"])
  empty = ConstantNet(MatrixStack,
                      requires_grad=opt["learn_constants"],
                      init=opt["init"])

  def ref_sketch(items, runstate):
    return trace(items, runstate, push=list_push, pop=list_pop, empty=list_empty)

  class StackSketch(asl.Sketch):
    def sketch(self, items, runstate):
      """Example stack trace"""
      return trace(items, runstate, push=push, pop=pop, empty=empty)

  stack_sketch = StackSketch([List[Mnist]], [Mnist])
  nstack = ModuleDict({"push": push,
                       "pop": pop,
                       "empty": empty,
                       "stack_sketch": stack_sketch})

  # Cuda that shit
  asl.cuda(nstack, opt["nocuda"])

  # Loss
  mnistiter = asl.util.mnistloader(opt["batch_size"])
  loss_gen = asl.single_ref_loss(stack_sketch,
                                 ref_sketch,
                                 mnistiter,
                                 refresh_mnist,
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
          "batch_size": [256, 512],
          "learn_constants": [True, False],
          "accum": [mean],
          "init": [torch.nn.init.uniform,
                   torch.nn.init.normal,
                   torch.nn.init.xavier_normal,
                   torch.nn.init.xavier_uniform,
                   torch.nn.init.kaiming_normal,
                   torch.nn.init.kaiming_uniform,
                   torch.ones_like,
                   torch.zeros_like],
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