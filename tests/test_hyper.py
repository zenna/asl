from asl.hyper.search import rand_local_hyper_search

def test_hyper():
  options = {'lr': [0.001, 0.1],
             'batch_size': [256, 512]}
  var_option_keys = ['lr', 'batch_size']
  filepath = "/home/zenna/repos/asl/tests/test_stack.py"
  nsamples = 2
  nrepeats = 1
  rand_local_hyper_search(filepath, options, var_option_keys, nsamples,
                          nrepeats=1, prefix='stack')
