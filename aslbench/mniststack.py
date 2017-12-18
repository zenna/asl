"Stack learned from reference"
from typing import List
import asl
from asl.modules.modules import ConstantNet, ModuleDict
from asl.structs.nstack import list_push, list_pop, list_empty
from torch import optim, nn
import common
from multipledispatch import dispatch

def parametric_sketch(items, push, pop, empty):
  """Example stack trace"""
  asl.log_append("empty", empty)
  stack = empty
  (stack,) = push(stack, next(items))
  asl.log_append("stack1", stack)
  asl.log_append("{}/internal".format(asl.mode().name), stack)
  (stack,) = push(stack, next(items))
  asl.log_append("{}/internal".format(asl.mode().name), stack)
  (pop_stack, pop_item) = pop(stack)
  asl.log_append("{}/internal".format(asl.mode().name), pop_stack)
  asl.observe(pop_item, "pop1")
  (pop_stack, pop_item) = pop(pop_stack)
  asl.log_append("{}/internal".format(asl.mode().name), pop_stack)
  asl.observe(pop_item, "pop2")
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

  push = Push(arch=opt.arch, arch_opt=opt.arch_opt)
  pop = Pop(arch=opt.arch, arch_opt=opt.arch_opt)
  empty = ConstantNet(MatrixStack)

  class StackSketch(asl.Sketch):
    def sketch(self, items):
      """Example stack trace"""
      return parametric_sketch(items, push=push, pop=pop, empty=empty)

  stack_sketch = StackSketch([List[Mnist]], [Mnist])
  asl.cuda(stack_sketch)
  asl.cuda(push)
  asl.cuda(pop)
  asl.cuda(empty)

  def ref_sketch(items):
    return parametric_sketch(items, push=list_push, pop=list_pop, empty=list_empty)

  # Loss
  mnistiter = asl.util.mnistloader(opt.batch_size)
  loss_gen = asl.ref_loss_gen(stack_sketch,
                              ref_sketch,
                              mnistiter,
                              lambda x: Mnist(asl.util.data.train_data(x)))

  # Optimization details
  parameters = list(push.parameters()) + list(pop.parameters()) + list(empty.parameters())
  optimizer = optim.Adam(parameters, lr=opt.lr)
  asl.opt.save_opt(opt)
  if opt.resume_path is not None and opt.resume_path != '':
    asl.load_checkpoint(opt.resume_path, nstack, optimizer)

  asl.train(loss_gen, optimizer, maxiters=100000,
        cont=asl.converged(1000),
        callbacks=[asl.print_loss(100),
                   common.plot_empty,
                   common.log_observes,
                   common.plot_observes,
                   common.plot_internals,
                   asl.save_checkpoint(1000, stack_sketch)
                   ],
        log_dir=opt.log_dir)


if __name__ == "__main__":
  opt = asl.handle_args(stack_args)
  opt = asl.handle_hyper(opt, __file__)
  if opt.sample:
    opt = asl.merge(stack_args_sample(), opt)
  asl.save_opt(opt)
  train_stack(opt)
