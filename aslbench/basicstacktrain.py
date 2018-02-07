import os
import random
import asl
from mniststack import train_stack, stack_args
import torch
from common import trainloadsave
from asl.loss import mean
from torch import optim
import torch.nn.functional as F

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


def optim_sampler():
  lr = 0.001
  optimizer = optim.Adam
  return {"optimizer": optimizer,
          "lr": lr}

def stack_optspace():
  arch_opt = {'batch_norm': True,
              'h_channels': 8,
              'nhlayers': 4,
              'activation': F.elu,
              'ks': 3,
              'last_activation': F.elu,
              'learn_batch_norm': True,
              'padding': 1}

  return {"tracegen": tracegen1,
          "nrounds": 1,
          "dataset": ["mnist"],
          "nchannels": 1,
          "nitems": 2,
          "normalize": [True],
          "batch_size": [16],
          "learn_constants": [True],
          "accum": mean,
          "init": [torch.nn.init.uniform],
          "arch_opt": arch_opt,
          "optim_args": optim_sampler}

def traces_gen(nsamples):
  # Delaying computation of this value because we dont know nsamples yet
  return asl.prodsample(stack_optspace(),
                        to_enum=[],
                        to_sample=["init",
                                   "batch_size",
                                   "lr",
                                   "learn_constants",
                                   "normalize"],
                        to_sample_merge=["optim_args"],
                        nsamples=nsamples)

if __name__ == "__main__":
  thisfile = os.path.abspath(__file__)
  res = trainloadsave(thisfile, train_stack, traces_gen, stack_args)