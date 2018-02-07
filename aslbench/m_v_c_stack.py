import asl
from mniststack import *
import torch
from stdargs import arch_sampler, optim_sampler
import common
import os
from asl.loss import mean


def mvc_arch_sampler():
  "Options sampler"
  arch = random.choice([#asl.archs.convnet.ConvNet,
                        asl.archs.mlp.MLPNet,
                        ])
  arch_opt = arch.sample_hyper(None, None)
  opt = {"arch": arch,
         "arch_opt": arch_opt}
  return opt

def sample_arch_opt(arch):
  return {"arch_opt": arch.sample_hyper(None, None)}

def stack_optspace():
  return {"nrounds": 1,
          "dataset": ["mnist"],
          "nchannels": 1,
          "nitems": 3,
          "normalize": [True, False],
          "batch_size": [16, 32, 64],
          "learn_constants": [True],
          "accum": mean,
          "init": [torch.nn.init.uniform,
                   torch.nn.init.normal,
                   torch.ones_like,
                   torch.zeros_like
                   ],
          "arch_opt": arch_sampler,
          "optim_args": optim_sampler,
          "arch": [asl.archs.convnet.ConvNet, asl.archs.mlp.MLPNet],
          "arch_opt": sample_arch_opt}

def mvc_gen(nsamples):
  # Delaying computation of this value because we dont know nsamples yet
  return asl.prodsample(stack_optspace(),
                        to_enum=["arch"],
                        to_sample=["init",
                                   "batch_size",
                                   "lr",
                                   "learn_constants",
                                   "normalize"],
                        to_sample_merge=["arch_opt", "optim_args"],
                        nsamples=nsamples)

if __name__ == "__main__":
  thisfile = os.path.abspath(__file__)
  res = common.trainloadsave(thisfile, train_stack, runoptsgen, stack_args)