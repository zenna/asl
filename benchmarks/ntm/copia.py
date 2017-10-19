"""Copy Benchmark - Can a neural network learn to copy
From NTM Paper:
- The network is presented with an input sequence of random binary vectors, followed by a delimiter flag.
- The networks were trained to copy sequences of eight bit random vectors, where the
sequence lengths were randomised between 1 and 20. The target sequence was simply a
copy of the input sequence (without the delimiter flag).
"""
from copy import copy
from functools import partial

from asl.sketch import Sketch, soft_ch
from asl.callbacks import every_n, print_loss
from asl.util.misc import cuda
from asl.util.io import handle_args
from asl.util.generators import infinite_samples
from asl.type import Type
from asl.structs.nqueue import EnqueueNet, DequeueNet, ref_queue
from asl.util.misc import iterget
from asl.train import train, max_iters
from asl.modules.modules import ConstantNet, ModuleDict
from asl.modules.archs import MLPNet
from asl.log import log_append
from torch import optim

from typing import List
import torch
from torch import nn
from torch.autograd import Variable

class CopySketch(Sketch):
  "Sketch for copy of list of elements"

  def __init__(self, element_type, model, ref_model, seq_len):
    super(CopySketch, self).__init__([List[element_type]],
                                     [List[element_type]],
                                     model,
                                     ref_model)
    self.choice_len = 10
    self.choice1 = nn.Parameter(torch.rand(1, self.choice_len), requires_grad=True)
    self.seq_len = seq_len

  def choice1f(self, outlen):
    proj = Variable(cuda(torch.rand(self.choice_len, outlen)))
    return torch.matmul(self.choice1, proj)
  # def sketch(self, items, enqueue, dequeue, empty):
  #   queue_chs = [empty]
  #   out_items = []
  #   item_chs = copy(items)
  #   for _ in range(self.seq_len):
  #     (queue,) = enqueue(soft_ch(queue_chs, self.choice1f),
  #                        soft_ch(item_chs, self.choice1f))
  #     queue_chs.append(queue)
  #
  #   for _ in range(self.seq_len):
  #     (queue, item) = dequeue(soft_ch(queue_chs, self.choice1f))
  #     queue_chs.append(queue)
  #     out_items.append(item)
  #
  #   return out_items
  def sketch(self, items, enqueue, dequeue, empty):
    queue_chs = [empty]
    out_items = []
    item_chs = copy(items)
    for i in range(self.seq_len):
      (queue,) = enqueue(queue_chs[-1],
                         items[i])
                        #  soft_ch(item_chs, self.choice1f))
      queue_chs.append(queue)

    for _ in range(self.seq_len):
      (queue, item) = dequeue(queue_chs[-1])
      queue_chs.append(queue)
      out_items.append(item)

    return out_items



def neural_queue(element_type, queue_type):
  enqueue_img = EnqueueNet(queue_type, element_type, arch=MLPNet)
  dequeue_img = DequeueNet(queue_type, element_type, MLPNet)
  empty_queue = ConstantNet(queue_type)
  neural_ref = ModuleDict({"enqueue": enqueue_img,
                           "dequeue": dequeue_img,
                           "empty": empty_queue})
  cuda(neural_ref)
  return neural_ref


def plot_items(i, log, writer, **kwargs):
  writer.add_image('fwd/1', log['outputs'][0][0][0], i)
  writer.add_image('fwd/2', log['outputs'][0][1][0], i)
  writer.add_image('fwd/3', log['outputs'][0][2][0], i)
  writer.add_image('rev/1', log['items'][0][0][0], i)
  writer.add_image('rev/2', log['items'][0][1][0], i)
  writer.add_image('rev/3', log['items'][0][2][0], i)


def benchmark_copy_sketch():
  opt = handle_args()
  string_len = 8

  class SeqStack(Type):
    size = (string_len, 1)

  class BernSeq(Type):
    size = (string_len, 1)

  def bern_eq(*shape):
    return cuda(torch.bernoulli(torch.ones(*shape).fill_(0.5)))

  seq_sampler = infinite_samples(bern_eq, opt.batch_size, (string_len, 1), True)
  nqueue = neural_queue(SeqStack, BernSeq)
  refqueue = ref_queue()
  seq_len = 3  # From paper: between 1 and 20
  copy_sketch = CopySketch(Type, nqueue, refqueue, seq_len)
  cuda(copy_sketch)

  def loss_gen():
    items = iterget(seq_sampler, seq_len)
    target_items = copy(items)
    outputs = copy_sketch(items)
    log_append("outputs", outputs)
    log_append("items", items)
    import pdb; pdb.set_trace()
    losses = [nn.BCELoss()(outputs[i], target_items[i]) for i in range(seq_len)]
    loss = sum(losses)
    print("LOSS", loss)
    return loss

  every = 100
  print_loss_gen = print_loss(every)
  next(print_loss_gen)

  optimizer = optim.Adam(copy_sketch.parameters(), opt.lr)
  print(opt)
  train(loss_gen, optimizer, cont=partial(max_iters, maxiters=1000000),
        callbacks=[every_n(plot_items, 100), print_loss_gen])

if __name__ == "__main__":
  benchmark_copy_sketch()
