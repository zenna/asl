"Stack learned from reference"
import random
import os
from typing import List
import asl
from asl.modules.modules import ConstantNet, ModuleDict
from asl.structs.nstack import list_push, list_pop, list_empty
from torch import optim, nn
import common
from asl.callbacks import every_n
from multipledispatch import dispatch
from tensorboardX import SummaryWriter

def tracegen(nitems, nrounds):
  print("Making trace with {} items".format(nitems))
  def trace(items, runstate, push, pop, empty):
    """Example stack trace"""
    asl.log_append("empty", empty)
    stack = empty
    for nr in range(nrounds):
      for i in range(nitems):
        (stack,) = push(stack, next(items))
        asl.log_append("{}/internal".format(runstate['mode']), stack)

      for j in range(nitems):
        (stack, pop_item) = pop(stack)
        asl.observe(pop_item, "pop.{}.{}".format(nr, j), runstate)
        asl.log_append("{}/internal".format(runstate['mode']), stack)
      
    return pop_item
  
  return trace

    # (stack,) = push(stack, next(items))
    # asl.log_append("{}/internal".format(runstate['mode']), stack)

    # (pop_stack, pop_item) = pop(stack)
    # asl.observe(pop_item, "pop1", runstate)
    # asl.log_append("{}/internal".format(runstate['mode']), pop_stack)

    # (pop_stack, pop_item) = pop(pop_stack)
    # asl.observe(pop_item, "pop2", runstate)
    # asl.log_append("{}/internal".format(runstate['mode']), pop_stack)

    # # Do one more push pop
    # (stack,) = push(pop_stack, next(items))
    # asl.log_append("{}/internal".format(runstate['mode']), stack)

    # (pop_stack, pop_item) = pop(stack)
    # asl.observe(pop_item, "pop3", runstate)
    # asl.log_append("{}/internal".format(runstate['mode']), pop_stack)


def stack_args(parser):
  # FIXME: Currently uunusued
  parser.add_argument('--nitems', type=int, default=3, metavar='NI',
                      help='number of iteems in trace (default: 3)')
  parser.add_argument('--nrounds', type=int, default=1, metavar='NR',
                      help='number of rounds in trace')

def stack_args_sample():
  "Options sampler"
  return argparse.Namespace(batch_norm=np.random.rand() > 0.5)

mnist_size = (1, 28, 28)

class MatrixStack(asl.Type):
  typesize = mnist_size

class Mnist(asl.Type):
  typesize = mnist_size

@dispatch(Mnist, Mnist)
def dist(x, y):
  return nn.MSELoss()(x.value, y.value)

def train_stack(opt):
  # arch = opt["arch"]
  # arch_opt = opt['']
  trace = tracegen(opt["nitems"], opt["nrounds"])
  class Push(asl.Function, asl.Net):
    def __init__(self, name="Push", **kwargs):
      asl.Function.__init__(self, [MatrixStack, Mnist], [MatrixStack])
      asl.Net.__init__(self, name, **kwargs)

  class Pop(asl.Function, asl.Net):
    def __init__(self, name="Pop", **kwargs):
      asl.Function.__init__(self, [MatrixStack], [MatrixStack, Mnist])
      asl.Net.__init__(self, name, **kwargs)

  push = Push(arch=opt["arch"], arch_opt=opt["arch_opt"])
  pop = Pop(arch=opt["arch"], arch_opt=opt["arch_opt"])
  empty = ConstantNet(MatrixStack)


  class StackSketch(asl.Sketch):
    def sketch(self, items, runstate):
      """Example stack trace"""
      return trace(items, runstate, push=push, pop=pop, empty=empty)

  # CUDA that shit
  stack_sketch = StackSketch([List[Mnist]], [Mnist])
  nstack = ModuleDict({"push": push,
                       "pop": pop,
                       "empty": empty,
                       "stack_sketch": stack_sketch})

  # Cuda that shit
  asl.cuda(nstack, opt["nocuda"])

  def ref_sketch(items, runstate):
    return trace(items, runstate, push=list_push, pop=list_pop, empty=list_empty)

  def refresh_mnist(dl):
    "Extract image data and convert tensor to Mnist data type"
    return [asl.refresh_iter(dl, lambda x: Mnist(asl.util.image_data(x)))]

  # Loss
  mnistiter = asl.util.mnistloader(opt["batch_size"])
  loss_gen = asl.single_ref_loss(stack_sketch,
                                 ref_sketch,
                                 mnistiter,
                                 refresh_mnist)

  # Setup optimizer
  parameters = list(push.parameters()) + list(pop.parameters()) + list(empty.parameters())
  optimizer = optim.Adam(parameters, lr=opt["lr"])
  # import pdb; pdb.set_trace()
  asl.opt.save_opt(opt)
  if opt["resume_path"] is not None and opt["resume_path"] != '':
    asl.load_checkpoint(opt["resume_path"], nstack, optimizer)

  tbkeys = ["batch_size", "lr", "name", "nitems", "batch_norm"]
  optstring = asl.hyper.search.linearizeoptrecur(opt, tbkeys)
  if opt["train"]:
    writer = SummaryWriter(os.path.join(opt["log_dir"], optstring))
    update_df, save_df = asl.callbacks.save_update_df(opt)
    asl.train(loss_gen,
              optimizer,
              # maxiters=10,
              cont=asl.converged(1000),
              callbacks=[asl.print_loss(1),
                        every_n(common.plot_empty, 10),
                        every_n(common.plot_observes, 10),
                        every_n(common.plot_internals, 10),
                        every_n(asl.save_checkpoint(nstack), 1000),
                        every_n(save_df, 100),
                        update_df],
              post_callbacks=[save_df],
              log_dir=opt["log_dir"],
              writer = writer)
  return nstack


def arch_sampler():
  "Options sampler"
  arch = random.choice([asl.archs.convnet.ConvNet,
                        #asl.archs.mlp.MLPNet,
                        ])
  arch_opt = arch.sample_hyper(None, None)
  opt = {"arch": arch,
         "arch_opt": arch_opt}
  return opt

def stack_optspace():
  return {"nrounds": [1, 2],
          "nitems": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 24, 48],
          "batch_size": [8, 16, 32, 64, 128],
          "arch_opt": arch_sampler,
          "lr": [0.01, 0.001, 0.0001, 0.00001]}

if __name__ == "__main__":
  # Add stack-specific parameters to the cmdlargs
  cmdrunopt, dispatch_opt = asl.handle_args(stack_args)
  if dispatch_opt["dispatch"]:
    morerunopts = asl.prodsample(stack_optspace(),
                                 to_enum=[],
                                 to_sample=["batch_size", "nitems", "lr", "nrounds"],
                                 to_sample_merge=["arch_opt"],
                                 nsamples=dispatch_opt["nsamples"])
    # Merge each runopt with command line opts (which take precedence)
    for opt in morerunopts:
      for k, v in cmdrunopt.items():
        if k not in opt:
          opt[k] = v
      # opt.update(cmdrunopt)

    thisfile = os.path.abspath(__file__)
    asl.dispatch_runs(thisfile, dispatch_opt, morerunopts)
  else:
    if dispatch_opt["optfile"] is not None:
      keys = None if cmdrunopt["resume_path"] is None else ["arch", "lr"] 
      cmdrunopt.update(asl.load_opt(cmdrunopt["optfile"], keys))
      print("Loaded", cmdrunopt)
      print("resume_path", cmdrunopt["resume_path"] is None)
      train_stack(cmdrunopt)
    else:
      asl.save_opt(cmdrunopt)
      train_stack(cmdrunopt)