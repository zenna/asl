"Stack learned from reference"
from typing import List
import asl
from asl.modules.modules import ConstantNet, ModuleDict
from asl.structs.nstack import list_push, list_pop, list_empty
from torch import optim, nn
import common
from multipledispatch import dispatch

def trace(items, runstate, push, pop, empty):
  """Example stack trace"""
  asl.log_append("empty", empty)
  stack = empty

  (stack,) = push(stack, next(items))
  asl.log_append("{}/internal".format(runstate['mode']), stack)

  (stack,) = push(stack, next(items))
  asl.log_append("{}/internal".format(runstate['mode']), stack)

  (pop_stack, pop_item) = pop(stack)
  asl.observe(pop_item, "pop1", runstate)
  asl.log_append("{}/internal".format(runstate['mode']), pop_stack)

  (pop_stack, pop_item) = pop(pop_stack)
  asl.observe(pop_item, "pop2", runstate)
  asl.log_append("{}/internal".format(runstate['mode']), pop_stack)

  # Do one more push pop
  (stack,) = push(pop_stack, next(items))
  asl.log_append("{}/internal".format(runstate['mode']), stack)

  (pop_stack, pop_item) = pop(stack)
  asl.observe(pop_item, "pop3", runstate)
  asl.log_append("{}/internal".format(runstate['mode']), pop_stack)

  return pop_item


def stack_args(parser):
  # FIXME: Currently uunusued
  parser.add_argument('--nitems', type=int, default=3, metavar='NI',
                      help='number of iteems in trace (default: 3)')
  parser.add_argument('--batch_norm', action='store_true', default=True,
                      help='Do batch norm')


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
  asl.cuda(stack_sketch, opt["nocuda"])
  asl.cuda(push, opt["nocuda"])
  asl.cuda(pop, opt["nocuda"])
  asl.cuda(empty, opt["nocuda"])

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

  # Optimization details
  parameters = list(push.parameters()) + list(pop.parameters()) + list(empty.parameters())
  print("LEARNING RATE", opt["lr"])
  optimizer = optim.Adam(parameters, lr=opt["lr"])
  asl.opt.save_opt(opt)
  if opt["resume_path"] is not None and opt["resume_path"] != '':
    asl.load_checkpoint(opt["resume_path"], nstack, optimizer)

  asl.train(loss_gen, optimizer, maxiters=100000,
        # cont=asl.converged(1000),
        callbacks=[asl.print_loss(100),
                   common.plot_empty,
                   common.plot_observes,
                   common.plot_internals,
                   asl.save_checkpoint(1000, stack_sketch)
                   ],
        log_dir=opt["log_dir"])


def stack_optspace():
  return {"num_items": [1, 2,],
          "batch_size": [1, 1]}

if __name__ == "__main__":
  # Add stack-specific parameters to the cmdlargs
  cmdrunopt, dispatch_opt = asl.handle_args(stack_args)
  if dispatch_opt["dispatch"]:
    morerunopts = asl.prodsample(stack_optspace(),
                                 to_enum=["num_items"],
                                 to_sample=["batch_size"],
                                 nsamples=dispatch_opt["nsamples"])
    # Merge each runopt with command line opts (which take precedence)
    for opt in morerunopts:
      # FIXME: merging is wrong
      opt.update(cmdrunopt)

    asl.dispatch_runs(__file__, dispatch_opt, morerunopts)
  else:
    if dispatch_opt["optfile"] is not None:
      opt = asl.load_opt(dispatch_opt["optfile"])
      train_stack(opt)
    else:
      asl.save_opt(cmdrunopt)
      train_stack(cmdrunopt)