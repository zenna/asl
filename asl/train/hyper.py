import os
import subprocess


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
