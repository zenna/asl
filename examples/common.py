import os

def default_benchmark_options():
    "Get default options for asl training"
    options = {}
    options['num_iterations'] = (int, 1000)
    options['save_every'] = (int, 100)
    options['batch_size'] = (int, 512)
    options['dirname'] = (str, "dirname")
    options['datadir'] = (str, os.path.join(os.environ['DATADIR'], "asl"))
    return options
