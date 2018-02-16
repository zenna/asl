"Options"
import os
import sys
import argparse
from argparse import Namespace
import pprint
import asl
import asl.util.io
from asl.hyper.search import run_local_batch, run_sbatch, run_local_chunk, run_sbatch_chunk
from random import choice
from torch import optim
import torch
import itertools
from typing import Callable, Any, Iterable
from multipledispatch import dispatch
import subprocess
import pprint

# Options = Dict{Symbol, Any} # FIXME: reprecate in place of rundata

@dispatch(Iterable)
def sample(x):
  return choice(x)

@dispatch(Callable)
def sample(f):
  return f()

def prodsample(optspace, to_enum=[], to_sample=[], to_sample_merge=[], nsamples = 1):
  """Enumerate product space of `to_enum` sampling from `tosampale`
  # Arguments
  to_enum: Will enumerate through cartesian product of all elements of to_enum
  to_sample: Will sample from and put result in dictionary
  to_sample_merge: Sampling from should return a subdict to be merged

  >>> import random
  >>> optspace = {"names" : [1,2,3,4], "sally" : [2,4,5], "who": random.random}>>> to_enum = 'names'
  >>> to_enum = ["names"]
  >>> to_sample = ["who"]
  >>> prodsample(optspace, to_enum, to_sample)
  [{'who': 0.7574782819980577, 'names': 1, 'sally': [2, 4, 5]}, {'who': 0.7256589552411237, 'names': 2, 'sally': [2, 4, 5]}, {'who': 0.5548214067039823, 'names': 3, 'sally': [2, 4, 5]}, {'who': 0.04731431826541499, 'names': 4, 'sally': [2, 4, 5]}]  
  """
  to_enumprod = {k: optspace[k] for k in optspace.keys() if k in to_enum}
  to_sample = {k: optspace[k] for k in optspace.keys() if k in to_sample}
  iter = itertools.product(*to_enumprod.values())
  dicts = []
  for it in iter:
    subdict1 = dict(zip(to_enumprod.keys(), it))
    for i in range(nsamples):
      subdict2 = {k : sample(optspace[k]) for k in to_sample.keys()}
      subdict = optspace.copy()
      subdict.update(subdict1)
      subdict.update(subdict2)

      for k in to_sample_merge:
        sd = optspace[k]()
        subdict.update(sd)
      dicts.append(subdict)
  
  return dicts


def opt_as_string(opt):
  "Options as a string"
  return pprint.pformat(opt, indent=4)


def std_opt_sampler():
  "Options sampler"
  # Generic Options
  batch_size = choice([32, 64, 96, 128])
  lr = choice([0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1])
  optim_algo = choice([optim.Adam])
  arch = choice([asl.archs.convnet.ConvNet])
  arch_opt = arch.sample_hyper(None, None)

  opt = Namespace(hyper=False,
                  sample=True,
                  batch_size=batch_size,
                  test_batch_size=batch_size,
                  optim_algo=optim_algo,
                  lr=lr,
                  arch=arch,
                  arch_opt=arch_opt)
  return opt


def add_std_args(parser):
  # Run Args
  parser.add_argument('--train', action='store_true', default=False,
                    help='Train the model')
  parser.add_argument('--name', type=str, default='', metavar='JN',
                      help='Name of job')
  parser.add_argument('--group', type=str, default='', metavar='JN',
                      help='Group name')
  parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                      help='input batch size for training (default: 64)')
  parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                      help='input batch size for testing (default: 1000)')
  parser.add_argument('--epochs', type=int, default=10, metavar='N',
                      help='number of epochs to train (default: 10)')
  parser.add_argument('--log_dir', type=str, metavar='D',
                      help='Path to store data')
  parser.add_argument('--resume_path', type=str, default=None, metavar='R',
                      help='Path to resume parameters from')
  parser.add_argument('--nocuda', action='store_true', default=False,
                      help='disables CUDA training')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 1)')
  parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                      help='learning rate (default: 0.01)')

def add_dispatch_args(parser):
  # Dispatch args
  parser.add_argument('--dispatch', action='store_true', default=False,
                      help='Dispatch many jobs')
  parser.add_argument('--sample', action='store_true', default=False,
                    help='Sample parameter values')
  parser.add_argument('--optfile', type=str, default=None,
                    help='Specify load file to get options from')
  parser.add_argument('--jobsinchunk', type=int, default=1, metavar='C',
                      help='Jobs to run per machine (default 1)')
  parser.add_argument('--nsamples', type=int, default=1, metavar='NS',
                      help='number of samples for hyperparameters (default: 10)')
  parser.add_argument('--blocking', action='store_true', default=True,
                      help='Is hyper parameter search blocking?')
  parser.add_argument('--slurm', action='store_true', default=False,
                      help='Use the SLURM batching system')
  parser.add_argument('--dryrun', action='store_true', default=False,
                    help='Do a dry run, does not call subprocess')

def handle_log_dir(opt):
  # if log_dir was specified, just keep that
  # if log_dir not specified and name or group is specifeid
  if opt["log_dir"] is None:
    opt["log_dir"] = asl.util.io.log_dir(group=opt["group"], comment=opt["name"])


def handle_cuda(opt):
  if not opt["nocuda"] and not torch.cuda.is_available():
    print("Chose CUDA but CUDA not available, continuing without CUDA!")
    opt["cuda"] = False


def handle_arch(opt):
  opt["arch"] = asl.archs.convnet.ConvNet
  opt["arch_opt"] = {}


def add_git_info(opt):
  label = subprocess.check_output(["git", "describe", "--always"]).strip()
  opt["git_commit"] = label


def handle_args(*add_cust_parses):
  "Handle command liner arguments"
  # add_cust_parses modifies the parses to add custom arguments
  parser = argparse.ArgumentParser(description='')
  add_std_args(parser)
  add_dispatch_args(parser)
  for add_cust_parse in add_cust_parses:
    add_cust_parse(parser)
  run_opt = parser.parse_args().__dict__

  # handle_log_dir
  handle_log_dir(run_opt)
  handle_cuda(run_opt)
  handle_arch(run_opt)
  add_git_info(run_opt)
  # dispatch_opt = run_opt[]
  dargs = ["dispatch", "dryrun", "sample", "optfile", "jobsinchunk", "nsamples", "blocking", "slurm"]
  dispatch_opt = {k : run_opt[k] for k in dargs}
  return run_opt, dispatch_opt


def merge(opt1, opt2):
  "Merge opts, opt1 takes precedence"
  opt = Namespace()
  optv = vars(opt)
  for k, v in opt2._get_kwargs():
    optv[k] = v

  for k, v in opt1._get_kwargs():
    optv[k] = v

  return opt

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def dispatch_runs(runpath, dispatch_opt, runopts):
  # Split up the jobs into sets and dispatch
  jobchunks = chunks(runopts, dispatch_opt["jobsinchunk"])
  i = 0
  for chunk in jobchunks:
    i = i + 1
    if dispatch_opt["slurm"]:
      run_sbatch_chunk(runpath, chunk, dryrun=dispatch_opt["dryrun"])
    else:
      run_local_chunk(runpath, chunk, blocking=dispatch_opt["blocking"],
                      dryrun=dispatch_opt["dryrun"])
    print("Dispatched {} chunks".format(i))




def isopt(d):
  "Is dictionary d an opt"
  return type(d) is dict and all((type(k) is str for k in d.keys()))


def sanitize_opt(opt, conv):
  "Convert opt into something that can be pickled"
  newopt = {}
  for k, v in opt.items():
    # import pdb; pdb.set_trace()
    if isopt(v):
      newopt[k] = sanitize_opt(v, conv)
    elif v in list(conv.keys()): # ugh
      newopt[k] = conv[v]
    else:
      newopt[k] = opt[k]
  return newopt


def load_opt(path, keys=None):
  "Load options file from disk, optionally restrict to keys"
  opt = torch.load(path)
  opt = sanitize_opt(opt, asl.util.misc.STRINGTOF)
  if keys is None:
    return opt
  else:
    return {k: opt[k] for k in keys}


def save_opt(opt, savepath=None):
  opt = sanitize_opt(opt, asl.util.misc.FTOSTRING)
  if savepath is None:
    savepath = opt["log_dir"]

  "Save an options file to disk, as well as human readable .txt equivalent"
  asl.util.io.directory_check(savepath)
  print("Saving Options to ", savepath)
  pprint.PrettyPrinter().pprint(opt)
  savefullpath = os.path.join(savepath, "opt.pkl")
  torch.save(opt, savefullpath)
  torch.save(opt_as_string(opt), os.path.join(savepath, "optstring.txt"))
  return savefullpath
