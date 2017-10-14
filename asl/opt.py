"Options"
import pprint
from collections import namedtuple
import asl
from numpy.random import choice
from torch import optim
import numpy as np

Opt = namedtuple('Opt', ['log_dir',
                         'resume_path',
                         'batch_size',
                         'lr',
                         'optim',
                         'template',
                         'template_opt',
                         'specific'],
                 verbose=False)


def opt_as_string(opt):
  return pprint.pformat(opt._asdict(), indent=4)


def std_opt_gen(specific=None):
  "Options sampler"
  # Generic Options
  log_dir = asl.util.io.log_dir(group="mnistqueue")
  asl.util.io.directory_check(log_dir)
  batch_size = choice([32, 64, 96, 128])
  lr = choice([0.0001, 0.001, 0.01, 0.1])
  optim_algo = choice([optim.Adam])
  template = choice([asl.modules.templates.VarConvNet])

  template_opt = {}
  if template == asl.modules.templates.VarConvNet:
    template_opt['nlayers'] = choice([1])
    template_opt['batch_norm'] = np.random.rand() > 0.99

  opt = Opt(log_dir,
            None,
            batch_size,
            lr,
            optim_algo,
            template,
            template_opt,
            specific)
  return opt
