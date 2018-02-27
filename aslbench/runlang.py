import os
import asl
from lang import train_clevrlang, lang_args
import torch
import common
import numpy as np
import random
import torch.nn.functional as F
from torch import optim
from torchvision.datasets import MNIST
from asl.datasets import Omniglot
from asl.archs import NormalizeNet, CombineNet, SimpleConvNet
import torch.nn as nn

## Hyper Params
## ============
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
  last_act = random.choice([F.sigmoid])
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

## Network architectures
## ====================
def describe_arch():
  convparams = conv_hypers()
  def describe_arch_(in_sizes, output_sizes, **kwargs):
    import pdb; pdb.set_trace()
    sentence_size = output_sizes[0]
    nets = [NormalizeNet(in_sizes, out_sizes),
                         CombineNet(),
                         SimpleConvNet(**convparams),
                         nn.Linear(),
                         nn.Softmax]
    return nn.Sequential(nets)
  return describe_arch_

def which_image_arch():
  convparams = conv_hypers()
  def which_image_arch_(in_sizes, output_sizes, **kwargs):
    nets = [NormalizeNet(in_sizes, out_sizes),
                         CombineNet(),
                         SimpleConvNet(**convparams),
                         nn.Softmax]
    return nn.Sequential(nets)
  return describe_arch_

def stack_optspace():
  return {"dataset": [MNIST, Omniglot],
          "nchannels": 1,
          "nimages": [3],
          "normalize": True,
          "batch_size": [16, 32, 64],
          "learn_constants": True,
          "accum": asl.loss.mean,
          "init": [torch.nn.init.uniform,
                   torch.nn.init.normal],
          "describe_arch": describe_arch,
          "which_image_arch": which_image_arch,
          "optim_args": optim_sampler}

def lang_gen(nsamples):
  # Delaying computation of this value because we dont know nsamples yet
  return asl.prodsample(stack_optspace(),
                        to_sample=["dataset",
                                   "nimages",
                                   "init",
                                   "batch_size",
                                   "describe_arch",
                                   "which_module_arch"],
                        to_sample_merge=["optim_args"],
                        nsamples=nsamples)

if __name__ == "__main__":
  thisfile = os.path.abspath(__file__)
  res = common.trainloadsave(thisfile, train_clevrlang, lang_gen, lang_args)