"Stack learned from reference"
from typing import List
import asl
from asl.modules.modules import ConstantNet, ModuleDict
from asl.structs.nstack import ref_stack
from torch import optim, nn
import common
from multipledispatch import dispatch

class StackSketch(asl.Sketch):
  def sketch(self, items, push, pop, empty):
    """Example stack trace"""
    asl.log_append("empty", empty)
    stack = empty
    (stack,) = push(stack, next(items))
    (stack,) = push(stack, next(items))
    (pop_stack, pop_item) = pop(stack)
    self.observe(pop_item)
    (pop_stack, pop_item) = pop(pop_stack)
    self.observe(pop_item)
    return pop_item

def mnist_args(parser):
  parser.add_argument('--nitems', type=int, default=3, metavar='NI',
                      help='number of iteems in trace (default: 3)')

mnist_size = (1, 28, 28)

class MatrixStack(asl.Type):
  typesize = mnist_size

class Mnist(asl.Type):
  typesize = mnist_size

@dispatch(Mnist, Mnist)
def dist(x, y):
  return nn.MSELoss()(x.value, y.value)

def train_stack():
  # Get options from command line
  opt = asl.opt.handle_args(mnist_args)
  opt = asl.opt.handle_hyper(opt, __file__)

  class Push(asl.Function, asl.Net):
    def __init__(self="Push", name="Push", **kwargs):
      asl.Function.__init__(self, [MatrixStack, Mnist], [MatrixStack])
      asl.Net.__init__(self, name, **kwargs)

  class Pop(asl.Function, asl.Net):
    def __init__(self="Pop", name="Pop", **kwargs):
      asl.Function.__init__(self, [MatrixStack], [MatrixStack, Mnist])
      asl.Net.__init__(self, name, **kwargs)

  nstack = ModuleDict({'push': Push(arch=opt.arch,
                                    arch_opt=opt.arch_opt),
                       'pop': Pop(arch=opt.arch,
                                  arch_opt=opt.arch_opt),
                       'empty': ConstantNet(MatrixStack)})

  stack_sketch = StackSketch([List[Mnist]], [Mnist], nstack, ref_stack())
  asl.cuda(stack_sketch)

  # Loss
  mnistiter = asl.util.mnistloader(opt.batch_size)
  loss_gen = asl.sketch.loss_gen_gen(stack_sketch,
                                     mnistiter,
                                     lambda x: Mnist(asl.util.data.train_data(x)))

  # Optimization details
  optimizer = optim.Adam(nstack.parameters(), lr=opt.lr)
  asl.opt.save_opt(opt)
  if opt.resume_path is not None and opt.resume_path != '':
    asl.load_checkpoint(opt.resume_path, nstack, optimizer)

  asl.train(loss_gen, optimizer, maxiters=100000,
        cont=asl.converged(1000),
        callbacks=[asl.print_loss(100),
                   common.plot_empty,
                   common.plot_observes,
                   asl.save_checkpoint(1000, nstack)],
        log_dir=opt.log_dir)


if __name__ == "__main__":
  train_stack()
