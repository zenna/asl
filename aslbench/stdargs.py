import random
from torch import optim
import asl

def optim_sampler():
  # lr = random.choice([0.01, 0.001, 0.0001, 0.00001])
  lr = random.choice([0.001, 0.0001, 0.00001])
  # lr = 0.001

  optimizer = random.choice([optim.Adam])
  return {"optimizer": optimizer,
          "lr": lr}

def arch_sampler():
  "Options sampler"
  arch = random.choice([#asl.archs.convnet.ConvNet,
                        asl.archs.mlp.MLPNet,
                        ])
  arch_opt = arch.sample_hyper(None, None)
  opt = {"arch": arch,
         "arch_opt": arch_opt}
  return opt