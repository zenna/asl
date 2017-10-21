import asl
from aslbench.clevr.clevrsketch import benchmark_clevr_sketch, clevr_args, clevr_args_sample

if __name__ == "__main__":
  opt = asl.handle_args(clevr_args)
  opt = asl.handle_hyper(opt, __file__)
  if opt.sample:
    opt = asl.merge(clevr_args_sample(), opt)
  asl.save_opt(opt)
  benchmark_clevr_sketch(**vars(opt))
