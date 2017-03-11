from wacacore.train.hyper import rand_product
import os
import subprocess

def make_batch_string(options):
    batch_string = ["--%s=%s" % (k, v) for k, v in options.items()]
    return batch_string


def run_sbatch(options):
    run_str = ['sbatch', 'run.sh'] + make_batch_string(options)
    print(run_str)
    subprocess.call(run_str)


def hyper_search():
    options = {'train': True,
               'save': True,
               'num_iterations': 100000,
               'save_every':1000,
               'batch_size': [256, 512],
               'datadir': os.path.join(os.environ['DATADIR'], "pdt"),
               'nblocks': [1, 2, 3, 4],
               'block_size': [1, 2, 3, 4],
            #    'field_shape': [(10,), (50,), (100,), (500,), (1000,)],
               'nl': ['relu', 'elu']}
    var_option_keys = ['nblocks',
                       'block_size',
                       'batch_size',
                    #    'field_shape',
                       'nl']
    rand_product(run_sbatch, options, var_option_keys, 2, nrepeats=1, prefix='scalarfieldf')

if __name__ == "__main__":
    hyper_search()
