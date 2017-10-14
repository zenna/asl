import sys
import argparse
import torch
import os
import socket
from datetime import datetime
import asl


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
  parser.add_argument('--dirname', type=str, default='', metavar='D',
                      help='Path to store data')
  parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                      help='learning rate (default: 0.01)')
  parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                      help='SGD momentum (default: 0.5)')
  parser.add_argument('--no-cuda', action='store_true', default=False,
                      help='disables CUDA training')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 1)')
  parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                      help='how many batches to wait before logging training status')

def add_hyper_params(parser):
  parser.add_argument('--nsamples', type=int, default=10, metavar='NS',
                      help='number of samples for hyperparameters (default: 10)')


def handle_args(*add_cust_parses):
  parser = argparse.ArgumentParser(description='')
  add_std_args(parser)
  for add_cust_parse in add_cust_parses:
    add_cust_parse(parser)
  args = parser.parse_args()
  args.cuda = not args.no_cuda and torch.cuda.is_available()
  return args


def handle_hyper(path, opt_sampler):
  # println("Show PATH", path)
  opt = handle_args()
  if opt.hyper:
    print("Doing Hyper Parameter Search")
    opt = handle_args(add_hyper_params)
    for _ in range(opt.nsamples):
      opt_dict = {'sample': True}
      asl.hyper.search.run_local_batch(path, opt_dict, blocking=True)
    sys.exit()
  if opt.sample:
    print("Sampling opt values from sampler")
    opt = opt_sampler()
  print(opt)
  return opt


def datadir(default='./data', varname='DATADIR'):
  "Data directory"
  if varname in os.environ:
    return os.environ['DATADIR']
  else:
    return default


def log_dir(root=datadir(), group='nogroup', comment=''):
  "Log directory, e.g. ~/datadir/mnist/Oct14_02-43-22_my_comp/"
  return os.path.join(root,
                      'runs',
                      group,
                      datetime.now().strftime('%b%d_%H-%M-%S')+'_'+socket.gethostname()+comment)


def directory_check(path):
  '''Initialize the directory for log files.'''
  # If the direcotry does not exist, create it!
  if not os.path.exists(path):
      os.makedirs(path)
