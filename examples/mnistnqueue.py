import sys
import os
import asl
from asl.opt import  opt_as_string
from asl.structs.nstack import PushNet, PopNet
from asl.modules.modules import ConstantNet, ModuleDict
from asl.util.misc import cuda
from asl.type import Type
from asl.callbacks import print_loss, converged, save_checkpoint, load_checkpoint
from asl.util.misc import iterget, train_data
from asl.util.io import handle_args
from asl.util.data import trainloader
from asl.log import log_append
from asl.train import train
from asl.structs.nstack import ref_stack
from asl.loss import observe_loss
from asl.hyper.search import run_local_batch
from numpy.random import choice
import torch
from torch import optim, nn


# Make function of nitems
def stack_trace(items, push, pop, empty):
  """Example stack trace"""
  log_append("empty", empty)

  observes = []
  stack = empty
  # print("mean", items[0].mean())
  (stack,) = push(stack, items[0])
  (stack,) = push(stack, items[1])
  # (stack,) = push(stack, items[2])
  (pop_stack, pop_item) = pop(stack)
  observes.append(pop_item)
  (pop_stack, pop_item) = pop(pop_stack)
  observes.append(pop_item)
  log_append("observes", observes)
  log_append("empty", empty)
  return observes


def plot_observes(i, log, writer, **kwargs):
  "Show the empty set in tensorboardX"
  refobserves = log['observes'][0]
  nobserves = log['observes'][1]
  for j in range(len(refobserves)):
    writer.add_image('compare{}/ref'.format(j), refobserves[j][0], i)
    writer.add_image('compare{}/neural'.format(j), nobserves[j][0], i)


def plot_empty(i, log, writer, **kwargs):
  "Show the empty set in tensorboardX"
  img = log['empty'][0].value
  writer.add_image('EmptySet', img, i)


def train_stack():
  opt = asl.util.io.handle_hyper(__file__, lambda: asl.opt.std_opt_gen({'nitems': 3}))

  nitems = opt.specific['nitems']
  mnist_size = (1, 28, 28)

  class MatrixStack(Type):
    size = mnist_size

  class Mnist(Type):
    size = mnist_size

  tl = trainloader(opt.batch_size)
  items_iter = iter(tl)
  ref_items_iter = iter(tl)
  nstack = ModuleDict({'push': PushNet(MatrixStack, Mnist, template=opt.template, template_opt=opt.template_opt),
                       'pop': PopNet(MatrixStack, Mnist, template=opt.template, template_opt=opt.template_opt),
                       'empty': ConstantNet(MatrixStack)})
  cuda(nstack)
  refstack = ref_stack()
  criterion = nn.MSELoss()
  optimizer = optim.Adam(nstack.parameters(), lr=opt.lr)

  def loss_gen():
    nonlocal items_iter, ref_items_iter

    try:
      items = iterget(items_iter, nitems, transform=train_data)
      ref_items = iterget(ref_items_iter, nitems, transform=train_data)
    except StopIteration:
      print("End of Epoch")
      items_iter = iter(tl)
      ref_items_iter = iter(tl)
      items = iterget(items_iter, nitems, transform=train_data)
      ref_items = iterget(ref_items_iter, nitems, transform=train_data)

    observes = stack_trace(items, **nstack)
    refobserves = stack_trace(ref_items, **refstack)
    return observe_loss(criterion, observes, refobserves)

  if opt.resume_path is not None:
    load_checkpoint(opt.resume_path, nstack, optimizer)

  torch.save(opt, os.path.join(opt.log_dir, "opt.pkl"))
  torch.save(opt_as_string(opt), os.path.join(opt.log_dir, "optstring.txt"))

  train(loss_gen, optimizer, maxiters=100000,
        cont=converged(1000),
        callbacks=[print_loss(100),
                   plot_empty,
                   plot_observes,
                   save_checkpoint(1000, nstack)],
        log_dir=opt.log_dir)


if __name__ == "__main__":
  train_stack()
