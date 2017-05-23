import sys
from pdt.train_tf import *
from wacacore.util.io import handle_args
from tensortemplates.module import template_module, nl_module
import os

def default_benchmark_options():
    "Get default options for pdt training"
    options = {}
    options['num_iterations'] = (int, 1000)
    options['save_every'] = (int, 100)
    options['batch_size'] = (int, 512)
    options['dirname'] = (str, "dirname")
    options['datadir'] = (str, os.path.join(os.environ['DATADIR'], "pdt"))
    return options


def game_options(adt):
    options = {}
    if adt == 'atari':
        options['game'] = (str, 'Breakout-v0')
    return options
