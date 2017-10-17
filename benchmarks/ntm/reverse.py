"""Reverse Benchmark - Can a neural network learn to reverse a sequence"""
from benchmarks.types import vec_stack, bern_seq, matrix_stack
import benchmarks.common as common
import asl.opt

from asl.templates.packing import stretch_cat
from asl.templates.convnet import VarConvNet
from asl.templates.mlp import MLPNet
from asl.sketch import Sketch, soft_ch
from asl.callbacks import every_n, print_loss, converged, save_checkpoint
from asl.util.misc import cuda
from asl.opt import handle_hyper
from asl.util.generators import infinite_samples
from asl.type import Type
from asl.structs.nstack import PushNet, PopNet, ref_stack
from asl.util.misc import iterget, take
from asl.train import train, max_iters
from asl.modules.modules import ConstantNet, ModuleDict
from asl.log import log_append, log
from asl.loss import vec_dist
from torch import optim

from typing import List
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

# TODO
# - Get the template sampling
# - Update the sketch so that it is possible
# - Make different modes for differnet kinds of training
   # opt both ovserve loss and supervised loss
   # opt just supervised loss
   # if doing both, minimize both functiosn
   # minimize one after the other
   # minimize one in each step


def onehot(i, onehot_len, batch_size):
  # Dummy input that HAS to be 2D for the scatter (you can use view(-1,1) if needed)
  y = torch.LongTensor(batch_size, 1).fill_(i)
  # One hot encoding buffer that you create out of the loop and just keep reusing
  y_onehot = torch.FloatTensor(batch_size, onehot_len)
  # In your for loop
  y_onehot.zero_()
  return Variable(cuda(y_onehot.scatter_(1, y, 1)), requires_grad=False)


def plot_sketch(i, log, writer, batch=0, **kwargs):
  "Show internal structure"
  for (j, img) in enumerate(log['item_index']):
    writer.add_image('item_index/{}'.format(j), img, i)
  for (j, img) in enumerate(log['pop_item']):
    writer.add_image('pop_item/{}'.format(j), img[0:1, :], i)
  for (j, img) in enumerate(log['push_item']):
    writer.add_image('push_item/{}'.format(j), img[0:1, :], i)
  for (j, img) in enumerate(log['pop_stack']):
    writer.add_image('pop_stack/{}'.format(j), img[batch], i)
  for (j, img) in enumerate(log['push_stack']):
    writer.add_image('push_stack/{}'.format(j), img[batch], i)
  for (j, img) in enumerate(log['outputs']):
    writer.add_image('outputs/{}'.format(j), img[0:1, :], i)
  for (j, img) in enumerate(log['items']):
    writer.add_image('items/{}'.format(j), img[0:1, :], i)
  for (j, img) in enumerate(log['rev_items']):
    writer.add_image('rev_items/{}'.format(j), img[0:1, :], i)


class CopySketch(Sketch):
  "Sketch for copy of list of elements"

  def __init__(self, element_type, model, ref_model, seq_len):
    super(CopySketch, self).__init__([List[element_type]],
                                     [List[element_type]],
                                     model,
                                     ref_model)
    self.onehot_len = seq_len
    self.choosenet = MLPNet([(self.onehot_len,)], [(seq_len,)])
    self.seq_len = seq_len

  def choose_item(self, i):
    "Given i choose i"
    ionehot = onehot(i, self.onehot_len, 1)
    (item_choice, ) = self.choosenet(ionehot)
    return F.sigmoid(item_choice) # FIXME: Net should do this sigmoiding

  def sketch(self, items, push, pop, empty):
    # import pdb; pdb.set_trace()
    stack = empty
    out_items = []
    for i in range(self.seq_len):
      choice = log_append("item_index", self.choose_item(i))
      item = log_append("push_item", soft_ch(items, choice))
      (stack,) = push(stack, item)
      log_append("push_stack", stack)

    for i in range(self.seq_len):
      (stack, item) = pop(stack)
      log_append("pop_stack", stack)
      log_append("pop_item", item)
      out_items.append(item)

    return out_items

def plot_items(i, log, writer, **kwargs):
  writer.add_image('fwd/1', log['outputs'][0][0][0], i)
  writer.add_image('fwd/2', log['outputs'][0][1][0], i)
  writer.add_image('fwd/3', log['outputs'][0][2][0], i)
  writer.add_image('rev/1', log['items'][0][0][0], i)
  writer.add_image('rev/2', log['items'][0][1][0], i)
  writer.add_image('rev/3', log['items'][0][2][0], i)

def reverse_args(parser):
  parser.add_argument('--seq_len', type=int, default=4, metavar='NI',
                      help='Length of sequence')
  parser.add_argument('--stack_len', type=int, default=8, metavar='NI',
                      help='Length oitemsf sequence')


def benchmark_copy_sketch(batch_size, stack_len, seq_len, template, log_dir,
                          lr, template_opt, **kwargs):
  stack_len = stack_len
  seq_len = seq_len  # From paper: between 1 and 20
  BernSeq = bern_seq(seq_len)
  MatrixStack = matrix_stack(1, seq_len, seq_len)
  template_opt['combine_inputs'] = lambda xs: stretch_cat(xs,
                                                          MatrixStack.size,
                                                          2)
  template_opt['activation'] = F.sigmoid
  nstack = ModuleDict({'push': PushNet(MatrixStack, BernSeq,
                                             template=VarConvNet,
                                             template_opt=template_opt),
                       'pop': PopNet(MatrixStack, BernSeq,
                                             template=VarConvNet,
                                             template_opt=template_opt),
                       'empty': ConstantNet(MatrixStack)})

  refstack = ref_stack()
  copy_sketch = CopySketch(BernSeq, nstack, refstack, seq_len)
  cuda(copy_sketch)
  bern_iter = BernSeq.iter(batch_size)

  def loss_gen():
    # Should copy the sequence, therefore the output should
    items = take(bern_iter, seq_len)
    rev_items = items.copy()
    rev_items.reverse()
    outputs = copy_sketch(items)
    log("outputs", outputs)
    log("items", items)
    log("rev_items", rev_items)

    # import pdb; pdb.set_trace()
    return vec_dist(outputs, rev_items, dist=nn.BCELoss())

  optimizer = optim.Adam(copy_sketch.parameters(), lr)
  train(loss_gen,
        optimizer,
        cont=converged(1000),
        callbacks=[print_loss(100),
                   every_n(plot_sketch, 500),
                  #  common.plot_empty,
                  #  common.plot_observes,
                   save_checkpoint(1000, copy_sketch)],
        log_dir=log_dir)

if __name__ == "__main__":
  opt = asl.opt.handle_args(reverse_args)
  opt = asl.opt.handle_hyper(opt, __file__)
  asl.opt.save_opt(opt)
  benchmark_copy_sketch(**vars(opt))
