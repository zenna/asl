import asl
from mniststack import *
import torch
from stdargs import arch_sampler, optim_sampler
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

    # Final pop to make sure we get some data
    if stack_size > 0:
      (stack, pop_item) = pop(stack)
      asl.observe(pop_item, "pop.final", runstate)    
      asl.log_append("{}/internal".format(runstate['mode']), stack)

  return trace5


## Hyper Params
## ============

def stack_optspace():
  return {"tracegen": [tracegen1, tracegen2, tracegen3, tracegen4, tracegen5],
          "nrounds": [1, 2],
          "dataset": ["mnist"],
          "nchannels": 1,
          "nitems": 3,
          "normalize": [True],
          "batch_size": [16, 32, 64],
          "learn_constants": [True],
          "accum": mean,
          "init": [torch.nn.init.uniform,
                   torch.nn.init.normal],
          "arch_opt": arch_sampler,
          "optim_args": optim_sampler}

def traces_gen(nsamples):
  # Delaying computation of this value because we dont know nsamples yet
  return asl.prodsample(stack_optspace(),
                        to_enum=["tracegen"],
                        to_sample=["init",
                                   "batch_size",
                                   "lr",
                                   "normalize"],
                        to_sample_merge=["arch_opt", "optim_args"],
                        nsamples=nsamples)

if __name__ == "__main__":
  thisfile = os.path.abspath(__file__)
  res = common.trainloadsave(thisfile, train_stack, traces_gen, stack_args)