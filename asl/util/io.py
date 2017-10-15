import sys
import argparse
import os
import socket
from datetime import datetime


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
