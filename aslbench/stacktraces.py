import asl
from mniststack import *
import torch
import common
import os
from asl.loss import mean

## Traces
## ======

def tracegen1(nitems, nrounds):
  def trace1(items, r, runstate, push, pop, empty):
    """Push push push, pop pop pop"""
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
  
  return trace1


def tracegen2(nitems, nrounds):
  def trace2(items, r, runstate, push, pop, empty):
    """Push Pop Push Push Pop Pop Push Push Push Pop Pop Pop"""
    asl.log_append("empty", empty)
    stack = empty
    for nr in range(nrounds):
      for i in range(nitems):
        (stack,) = push(stack, next(items))
        asl.log_append("{}/internal".format(runstate['mode']), stack)
        pop_stack = stack
        for j in range(i, -1, -1):
          (pop_stack, pop_item) = pop(pop_stack)
          asl.log_append("{}/internal".format(runstate['mode']), pop_stack)
          asl.observe(pop_item, "pop.nr{}.i{}.j{}".format(nr, i, j), runstate)
    return pop_item
  
  return trace2


def tracegen3(nitems, nrounds):
  def trace3(items, r, runstate, push, pop, empty):
    """Pushes n items, to create n stacks, pops from random one"""
    asl.log_append("empty", empty)
    stack = empty
    for nr in range(nrounds):
      stacks = []
      for i in range(nitems):
        (stack, ) = push(stack, next(items))
        asl.log_append("{}/internal".format(runstate['mode']), stack)
        stacks.append(stack)

      stack = r.choice(stacks)
      asl.log_append("{}/internal".format(runstate['mode']), stack)
      (stack, pop_item) = pop(stack)
      asl.log_append("{}/internal".format(runstate['mode']), stack)
      asl.observe(pop_item, "pop.nr{}".format(nr), runstate)
  
  return trace3


def tracegen4(nitems, nrounds):
  def trace4(items, r, runstate, push, pop, empty):
    """Pushes n items, to create n stacks, pops from random one"""
    asl.log_append("empty", empty)
    stack = empty
    for nr in range(nrounds):
      stacks = []
      for i in range(nitems):
        (stack, ) = push(stack, next(items))
        asl.log_append("{}/internal".format(runstate['mode']), stack)
        stacks.append(stack)

      (stack, pop_item) = pop(stack)
      asl.observe(pop_item, "pop.nr{}.i{}".format(nr, i), runstate)
      asl.log_append("{}/internal".format(runstate['mode']), stack)

  return trace4


def tracegen5(nitems, nrounds):
  def trace5(items, r, runstate, push, pop, empty):
    """Make n random choices over whether to push or pop"""
    asl.log_append("empty", empty)
    stack = empty
    stack_size = 0
    choicesperround = nitems
    for nr in range(nrounds * choicesperround):
      if stack_size == 0:
        (stack, ) = push(stack, next(items))
        stack_size = stack_size + 1
      elif stack_size == nitems:
        (stack, pop_item) = pop(stack)
        asl.observe(pop_item, "pop.nr{}".format(nr), runstate)
        stack_size = stack_size - 1
      else:
        dopush = r.choice([True, False])
        if dopush:
          (stack, ) = push(stack, next(items))
          stack_size = stack_size + 1
        else:
          (stack, pop_item) = pop(stack)
          asl.observe(pop_item, "pop.nr{}".format(nr), runstate)
          stack_size = stack_size - 1
      asl.log_append("{}/internal".format(runstate['mode']), stack)


    # Final pop to make sure we get some data
    if stack_size > 0:
      (stack, pop_item) = pop(stack)
      asl.observe(pop_item, "pop.final", runstate)    
      asl.log_append("{}/internal".format(runstate['mode']), stack)

  return trace5


def tracegen6(nitems, nrounds):
  def trace6(items, r, runstate, push, pop, empty):
    """Pushes n items, to create n stacks, pops randnum times, then observe once"""
    asl.log_append("empty", empty)
    stack = empty
    for nr in range(nrounds):
      stacks = []
      for i in range(nitems):
        (stack, ) = push(stack, next(items))
        asl.log_append("{}/internal".format(runstate['mode']), stack)
        stacks.append(stack)

      npops = r.randint(1, nitems)
      for j in range(npops):
        (stack, pop_item) = pop(stack)
        asl.log_append("{}/internal".format(runstate['mode']), stack)
      
      asl.observe(pop_item, "pop.nr{}".format(nr), runstate)
  
  return trace6

## Hyper Params
## ============

import numpy as np
import random
import torch.nn.functional as F
from torch import optim

def optim_sampler():
  lr = random.choice([0.001, 0.0001, 0.00001])
  optimizer = random.choice([optim.Adam])
  return {"optimizer": optimizer,
          "lr": lr}

def conv_hypers(pbatch_norm=0.5, max_layers=6):
  "Sample hyper parameters"
  learn_batch_norm = np.random.rand() > 0.5
  nlayers = np.random.randint(2, max_layers)
  h_channels = random.choice([4, 8, 12, 16, 24])
  act = random.choice([F.elu])
  last_act = random.choice([F.elu])
  ks = random.choice([3, 5])
  arch_opt = {'batch_norm': True,
              'h_channels': h_channels,
              'nhlayers': nlayers,
              'activation': act,
              'ks': ks,
              'last_activation': last_act,
              'learn_batch_norm': learn_batch_norm,
              'padding': (ks - 1)//2}
  return {"arch": asl.archs.convnet.ConvNet,
          "arch_opt": arch_opt}

def stack_optspace():
  return {"tracegen": [tracegen1, tracegen2, tracegen3, tracegen4, tracegen5, tracegen6],
          "nrounds": [2, 1],
          "dataset": ["mnist", "omniglot"],
          "nchannels": 1,
          "nitems": [3],
          "normalize": True,
          "batch_size": [16, 32, 64, 128],
          "learn_constants": True,
          "accum": mean,
          "init": [torch.nn.init.uniform,
                   torch.nn.init.normal],
          "arch_opt": conv_hypers,
          "optim_args": optim_sampler}

def traces_gen(nsamples):
  # Delaying computation of this value because we dont know nsamples yet
  return asl.prodsample(stack_optspace(),
                        to_enum=["tracegen", "dataset", "nrounds", "nitems"],
                        to_sample=["init",
                                   "batch_size"],
                        to_sample_merge=["arch_opt", "optim_args"],
                        nsamples=nsamples)

if __name__ == "__main__":
  thisfile = os.path.abspath(__file__)
  res = common.trainloadsave(thisfile, train_stack, traces_gen, stack_args)