import getopt
import os
import csv
import time
import sys
import argparse
import torch


def save_params(fname, params):
  f = open(fname, 'w')
  writer = csv.writer(f)
  for key, value in params.items():
    writer.writerow([key, value])
  f.close()


def save_dict_csv(fname, params):
  f = open(fname, 'w')
  writer = csv.writer(f)
  for key, value in params.items():
    writer.writerow([str(key), str(value)])
  f.close()


def append_time(sfx):
  return "%s%s" % (str(time.time()), sfx)


def gen_sfx_key(keys, options, add_time=True):
  sfx_dict = {}
  for key in keys:
    sfx_dict[key] = options[key]
  sfx = stringy_dict(sfx_dict)
  if add_time is True:
    sfx = append_time(sfx)
  print("sfx:", sfx)
  return sfx


def mk_dir(dirname, datadir=os.environ['DATADIR']):
  """Create directory with timestamp
  Args:
      sfx: a suffix string
      dirname:
      datadir: directory of all data
  """
  full_dir_name = os.path.join(datadir, dirname)
  print("Data will be saved to", full_dir_name)
  os.mkdir(full_dir_name)
  return full_dir_name


def add_std_args(parser):
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


def handle_args(*add_cust_parses):
  parser = argparse.ArgumentParser(description='')
  add_std_args(parser)
  for add_cust_parse in add_cust_parses:
    add_cust_parse(parser)
  args = parser.parse_args()
  args.cuda = not args.no_cuda and torch.cuda.is_available()
  return args
