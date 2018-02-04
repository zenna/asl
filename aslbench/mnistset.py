"Stack learned from reference"
import random
import os
from typing import List
import asl
from asl.modules.modules import ConstantNet, ModuleDict
from asl.structs.nstack import list_push, list_pop, list_empty
from torch import optim, nn
import common
from mnist import mnist_size, Mnist, dist, refresh_mnist
from asl.callbacks import every_n
from multipledispatch import dispatch
from tensorboardX import SummaryWriter

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
  empty = ConstantNet(MatrixStack)

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
                                 refresh_mnist)
  return common.trainmodel(opt, nstack, loss_gen)

## Samplers

def stack_args(parser):
  # FIXME: Currently uunusued
  parser.add_argument('--nitems', type=int, default=3, metavar='NI',
                      help='number of iteems in trace (default: 3)')
  parser.add_argument('--nrounds', type=int, default=1, metavar='NR',
                      help='number of rounds in trace')

def optim_sampler():
  lr = random.choice([0.01, 0.001, 0.0001, 0.00001])
  def gen_adam(params):
    return optim.Adam(params, lr=lr)

  optimizer = random.choice([gen_adam])
  return {"optimizer": optimizer,
          "lr": lr}

def arch_sampler():
  "Options sampler"
  arch = random.choice([asl.archs.convnet.ConvNet,
                        #asl.archs.mlp.MLPNet,
                        ])
  arch_opt = arch.sample_hyper(None, None)
  opt = {"arch": arch,
         "arch_opt": arch_opt}
  return opt

def stack_optspace():
  return {"nrounds": [1, 2],
          "nitems": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 24, 48],
          "batch_size": [8, 16, 32, 64, 128],
          "arch_opt": arch_sampler,
          "optim_args": optim_sampler}

def runoptsgen(nsamples):
  # Delaying computation of this value because we dont know nsamples yet
  return asl.prodsample(stack_optspace(),
                        to_enum=["nitems"],
                        to_sample=["batch_size", "lr", "nrounds"],
                        to_sample_merge=["arch_opt", "optim_args"],
                        nsamples=nsamples)

if __name__ == "__main__":
  res = common.trainloadsave(train_stack, runoptsgen, stack_args)