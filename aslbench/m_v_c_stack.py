import asl
from mniststack import *
import torch
import common
import os
from asl.loss import mean

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
  return {"tracegen": tracegen,
          "nrounds": 1,
          "dataset": ["mnist", 'omniglot'],
          "nchannels": 1,
          "nitems": 3,
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
                        to_enum=["dataset"],
                        to_sample=["init",
                                   "batch_size"],
                        to_sample_merge=["arch_opt", "optim_args"],
                        nsamples=nsamples)

if __name__ == "__main__":
  thisfile = os.path.abspath(__file__)
  res = common.trainloadsave(thisfile, train_stack, traces_gen, stack_args)