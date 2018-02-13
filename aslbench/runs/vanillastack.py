## Samplers

def stack_args(parser):
  parser.add_argument('--nitems', type=int, default=3, metavar='NI',
                      help='number of iteems in trace (default: 3)')
  parser.add_argument('--nrounds', type=int, default=1, metavar='NR',
                      help='number of rounds in trace')

def stack_optspace():
  return {"nrounds": [1, 2],
          "tracegen": tracegen,
          # "nitems": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20],
          "dataset": ["mnist", "omniglot"],
          "nchannels": 1,
          "nitems": [3],
          "normalize": [True, False],
          "batch_size": [16, 32, 64],
          "learn_constants": [True],
          "accum": [mean],
          "init": [torch.nn.init.uniform,
                   torch.nn.init.normal,
                   torch.ones_like,
                   torch.zeros_like
                   ],
          "arch_opt": arch_sampler,
          "optim_args": optim_sampler}

def runoptsgen(nsamples):
  # Delaying computation of this value because we dont know nsamples yet
  return asl.prodsample(stack_optspace(),
                        to_enum=["nitems", "dataset"],
                        to_sample=["init",
                                   "nrounds",
                                   "batch_size",
                                   "lr",
                                   "accum",
                                   "learn_constants",
                                   "normalize"],
                        to_sample_merge=["arch_opt", "optim_args"],
                        nsamples=nsamples)

if __name__ == "__main__":
  thisfile = os.path.abspath(__file__)
  res = common.trainloadsave(thisfile, train_stack, runoptsgen, stack_args)