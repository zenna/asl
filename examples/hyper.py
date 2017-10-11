import os
import subprocess
from wacacore.train.hyper import rand_product


def stringify(k, v):
  if v is True:
    return "--%s" % k
  elif v is False:
    return ""
  else:
    return "--%s=%s" % (k, v)


def make_batch_string(options):
  batch_string = [stringify(k, v) for k, v in options.items()]
  return batch_string


def run_sbatch(options):
  run_str = ['sbatch', 'run.sh'] + make_batch_string(options)
  print(run_str)
  subprocess.call(run_str)


def hyper_search():
  options = {'train': True,
             'save': True,
             'num_iterations': 100000,
             'save_every': 1000,
             'learning_rate': 0.001,
             'batch_size': [256, 512],
             'datadir': os.path.join(os.environ['DATADIR'], "asl"),
             'nblocks': [1, 2, 3, 4],
             'block_size': [1, 2, 3, 4],
             'nl': ['relu', 'elu']}
  var_option_keys = ['nblocks',
                     'block_size',
                     'batch_size',
                     'nl']
  rand_product(run_sbatch, options, var_option_keys, 10, nrepeats=1,
               prefix='scalarfieldf')

if __name__ == "__main__":
  hyper_search()
