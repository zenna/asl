"Options"
import os
import sys
import argparse
from argparse import Namespace
import pprint
import asl
import asl.util.io
from asl.hyper.search import run_local_batch, run_sbatch
from random import choice
from torch import optim
import torch


def opt_as_string(opt):
  "Options as a string"
  return pprint.pformat(vars(opt), indent=4)


def std_opt_sampler():
  "Options sampler"
  # Generic Options
  batch_size = choice([32, 64, 96, 128])
  lr = choice([0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1])
  optim_algo = choice([optim.Adam])
  template = choice([asl.templates.convnet.VarConvNet])
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


# Want
# Want to use default logdir but with group name that is taken from cmdline

def add_std_args(parser):
  parser.add_argument('--name', type=str, default='', metavar='JN',
                      help='Name of job')
  parser.add_argument('--group', type=str, default='', metavar='JN',
                      help='Group name')
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
  parser.add_argument('--log_dir', type=str, metavar='D',
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
  parser.add_argument('--slurm', action='store_true', default=False,
                      help='Use the SLURM batching system')

def handle_log_dir(opt):
  # if log_dir was specified, jsut keep that
  # if log_dir not specified and name or group is specifeid
  if opt.log_dir is None:
    opt.log_dir = asl.util.io.log_dir(group=opt.group, comment=opt.name)


def handle_cuda(opt):
  if opt.cuda and not torch.cuda.is_available():
    print("Chose CUDA but CUDA not available, continuing without CUDA!")
    opt.cuda = False

def handle_template(opt):
  opt.template = asl.templates.convnet.VarConvNet
  opt.template_opt = {}


def handle_args(*add_cust_parses):
  parser = argparse.ArgumentParser(description='')
  add_std_args(parser)
  add_hyper_params(parser)
  for add_cust_parse in add_cust_parses:
    add_cust_parse(parser)
  opt = parser.parse_args()

  handle_log_dir(opt)
  handle_cuda(opt)
  handle_template(opt)
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
    for _ in range(opt.nsamples):
      opt_dict = {'sample': True,
                  'name': opt.name,
                  'group': opt.group}
      if opt.slurm:
        # Add? --gres=gpu:1 --mem=16000
        sbatch_opt = {'job-name': opt.name,
                      'time': 720}
        run_sbatch(path, opt_dict, sbatch_opt)
      else:
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
