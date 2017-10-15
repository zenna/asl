"Options"
import os
import sys
import argparse
from argparse import Namespace
import pprint
import asl
from asl.hyper.search import run_local_batch
from numpy.random import choice
from torch import optim
import torch


def opt_as_string(opt):
  "Options as a string"
  return pprint.pformat(vars(opt), indent=4)


def std_opt_sampler():
  "Options sampler"
  # Generic Options
  batch_size = choice([32, 64, 96, 128])
  lr = choice([0.0001, 0.001, 0.01, 0.1])
  optim_algo = choice([optim.Adam])
  template = choice([asl.modules.templates.VarConvNet])
  template_opt = template.sample_hyper(None, None)

  opt = Namespace(hyper=False,
                  sample=True,
                  batch_size=batch_size,
                  test_batch_size=batch_size,
                  optim_algo=optim_algo,
                  lr=lr,
                  template=template,
                  template_opt=template_opt)
  return opt


def add_std_args(parser):
  parser.add_argument('--hyper', action='store_true', default=False,
                      help='Do hyper parameter search')
  parser.add_argument('--sample', action='store_true', default=False,
                      help='Sample parameter values')
  parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                      help='input batch size for training (default: 64)')
  parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                      help='input batch size for testing (default: 1000)')
  parser.add_argument('--epochs', type=int, default=10, metavar='N',
                      help='number of epochs to train (default: 10)')
  parser.add_argument('--log_dir', type=str, default=asl.util.io.log_dir(group="ungrouped"), metavar='D',
                      help='Path to store data')
  parser.add_argument('--resume_path', type=str, default='', metavar='R',
                      help='Path to resume parameters from')
  parser.add_argument('--cuda', action='store_true', default=True,
                      help='disables CUDA training')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 1)')
  parser.add_argument('--log_interval', type=int, default=10, metavar='LI',
                      help='how many batches to wait before logging training status')
  parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                      help='learning rate (default: 0.01)')


def add_hyper_params(parser):
  parser.add_argument('--nsamples', type=int, default=10, metavar='NS',
                      help='number of samples for hyperparameters (default: 10)')
  parser.add_argument('--blocking', action='store_true', default=False,
                      help='Is hyper parameter search blocking?')


def handle_args(*add_cust_parses):
  parser = argparse.ArgumentParser(description='')
  add_std_args(parser)
  for add_cust_parse in add_cust_parses:
    add_cust_parse(parser)
  opt = parser.parse_args()
  if opt.cuda and not torch.cuda.is_available():
    print("Chose CUDA but CUDA not available, continuing without CUDA!")
    opt.cuda = False

  opt.template = asl.modules.templates.VarConvNet
  opt.template_opt = {}
  return opt


def merge(opt1, opt2):
  "Merge opts, opt1 takes precedence"
  opt = Namespace()
  optv = vars(opt)
  for k, v in opt2._get_kwargs():
    optv[k] = v

  for k, v in opt1._get_kwargs():
    optv[k] = v

  return opt


def handle_hyper(opt, path, opt_sampler=std_opt_sampler):
  if opt.hyper:
    print("Starting hyper parameter search")
    opt = handle_args(add_hyper_params)
    for _ in range(opt.nsamples):
      opt_dict = {'sample': True}
      run_local_batch(path, opt_dict, blocking=True)
    sys.exit()
  if opt.sample:
    print("Sampling opt values from sampler")
    opt = merge(opt_sampler(), opt)
  return opt


def save_opt(opt):
    # Prepare directories
  asl.util.io.directory_check(opt.log_dir)
  print("Saving Options to ", opt.log_dir, "\n", opt)
  torch.save(opt, os.path.join(opt.log_dir, "opt.pkl"))
  torch.save(opt_as_string(opt), os.path.join(opt.log_dir, "optstring.txt"))
