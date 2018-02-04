import os
import socket
from datetime import datetime
import string
import random


def id_gen(size=4, chars=string.ascii_lowercase):
  return ''.join(random.choice(chars) for _ in range(size))


def datadir(default='./data', varname='DATADIR'):
  "Data directory"
  if varname in os.environ:
    return os.environ['DATADIR']
  else:
    return default

def log_dir_from_opt(opt):
  "Log directory, from run data"



def log_dir(root=datadir(), group='nogroup', comment='', id=id_gen()):
  "Log directory, e.g. ~/datadir/mnist/Oct14_02-43-22_my_comp/"
  fname = id + '_' + datetime.now().strftime('%b%d_%H-%M-%S')+'_'+socket.gethostname()+'_'+comment
  return os.path.join(root, 'runs', group, fname)


def directory_check(path):
  '''Initialize the directory for log files.'''
  # If the direcotry does not exist, create it!
  if not os.path.exists(path):
      os.makedirs(path)
