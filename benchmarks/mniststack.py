import os
import asl
from typing import List
from asl.opt import opt_as_string
from asl.structs.nstack import PushNet, PopNet
from asl.modules.modules import ConstantNet, ModuleDict
from asl.util.misc import cuda
from asl.type import Type
from asl.sketch import Sketch
from asl.callbacks import print_loss, converged, save_checkpoint, load_checkpoint
from asl.util.data import trainloader
from asl.log import log_append
from asl.train import train
from asl.structs.nstack import ref_stack
from numpy.random import choice
from torch import optim


def plot_observes(i, log, writer, **kwargs):
  "Show the empty set in tensorboardX"
  batch = 0
  for j in range(len(log['observes'])):
    writer.add_image('comp{}/ref'.format(j), log['observes'][j][batch], i)
    writer.add_image('comp{}/neural'.format(j), log['ref_observes'][j][batch], i)


def plot_empty(i, log, writer, **kwargs):
  "Show the empty set in tensorboardX"
  img = log['empty'][0].value
  writer.add_image('EmptySet', img, i)


class StackSketch(Sketch):
  def sketch(self, items, push, pop, empty):
    """Example stack trace"""
    log_append("empty", empty)
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

def train_stack():
  opt = asl.opt.handle_args(mnist_args)
  opt = asl.opt.handle_hyper(opt, __file__)
  nitems = 3
  mnist_size = (1, 28, 28)

  class MatrixStack(Type):
    size = mnist_size

  class Mnist(Type):
    size = mnist_size

  tl = trainloader(opt.batch_size)
  nstack = ModuleDict({'push': PushNet(MatrixStack, Mnist, arch=opt.arch, arch_opt=opt.arch_opt),
                       'pop': PopNet(MatrixStack, Mnist, arch=opt.arch, arch_opt=opt.arch_opt),
                       'empty': ConstantNet(MatrixStack)})

  stack_sketch = StackSketch([List[Mnist]], [Mnist], nstack, ref_stack())
  cuda(stack_sketch)
  loss_gen = asl.sketch.loss_gen_gen(stack_sketch, tl, asl.util.data.train_data)
  optimizer = optim.Adam(nstack.parameters(), lr=opt.lr)

  asl.opt.save_opt(opt)
  if opt.resume_path is not None and opt.resume_path != '':
    load_checkpoint(opt.resume_path, nstack, optimizer)

  train(loss_gen, optimizer, maxiters=100000,
        cont=converged(1000),
        callbacks=[print_loss(100),
                   plot_empty,
                   plot_observes,
                   save_checkpoint(1000, nstack)],
        log_dir=opt.log_dir)


if __name__ == "__main__":
  train_stack()
