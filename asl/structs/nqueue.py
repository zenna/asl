"Queue Data Structure trained from a reference implementation"
import itertools
from collections import deque

from asl.type import Type, Function, FunctionType, Constant
from asl.train import train
from asl.modules import VarConvNet
from asl.callbacks import tb_loss
from asl.util import draw, trainloader, as_img

import torch
from torch.autograd import Variable
import torch.nn as nn

import matplotlib.pyplot as plt


class Enqueue(Function):
  "Push Function for Queue"

  def __init__(self, queue_type, item_type):
    super(Enqueue, self).__init__([queue_type, item_type], [queue_type])
    # FIXME: Make these functiosn abstract and remove these selfs asignments
    self.queue_type = queue_type
    self.item_type = item_type


class Dequeue(Function):
  "Pop Function for Queue"

  def __init__(self, queue_type, item_type):
    super(Dequeue, self).__init__([queue_type], [queue_type, item_type])
    self.queue_type = queue_type
    self.item_type = item_type


class EnqueueNet(Enqueue, nn.Module):
  def __init__(self, queue_type, item_type):
    super(EnqueueNet, self).__init__(queue_type, item_type)
    self.module = VarConvNet(self.in_sizes(), self.out_sizes())

  def forward(self, x, y):
    return self.module.forward(x, y)


class DequeueNet(Dequeue, nn.Module):
  def __init__(self, queue_type, item_type):
    super(DequeueNet, self).__init__(queue_type, item_type)
    self.module = VarConvNet(self.in_sizes(), self.out_sizes())

  def forward(self, x):
    return self.module(x)


def list_enqueue(queue, element):
  queue = queue.copy()
  queue.append(element)
  return (queue, )


def list_dequeue(queue):
  queue = queue.copy()
  item = queue.popleft()
  return (queue, item)


def empty():
  return deque()


def queue_trace(items, enqueue, dequeue, empty):
  """Example queue trace"""
  items = [Variable(data[0].cuda()) for data in list(itertools.islice(items, 3))]
  # print("mean", [x.mean().data[0] for x in items])
  # if expand_empty:
  empty = empty()
  observes = []
  queue = empty
  # print("mean", items[0].mean())
  (queue,) = enqueue(queue, items[0])
  (queue,) = enqueue(queue, items[1])
  # (queue,) = enqueue(queue, items[2])
  (dequeue_queue, dequeue_item) = dequeue(queue)
  observes.append(dequeue_item)
  (dequeue_queue, dequeue_item) = dequeue(dequeue_queue)
  observes.append(dequeue_item)
  return observes


def plot_empty(i, model, reference, **kwargs):
  if i % 5000 == 0:
    draw(model["empty"].value)

def plot_compare(ax1, ax2, model_observes, ref_observes):
  nshow = 1
  for (i, _) in enumerate(model_observes):
    for j in range(nshow):
      im1 = as_img(model_observes[i][j])
      im2 = as_img(ref_observes[i][j])
      ax1.imshow(im1)
      ax2.imshow(im2)
      plt.draw()
      plt.pause(0.01)
      # import pdb; pdb.set_trace()

def show_ds(tl):
  """Example queue trace"""
  f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
  ax1.set_title('Compare')

  def plot_batch(i, model, reference, **kwargs):
    if i % 5000 == 0:
      items = iter(tl)
      model_observes = queue_trace(items, **model)
      ref_observes = queue_trace(items, **reference)
      plot_compare(ax1, ax2, model_observes, ref_observes)

  return plot_batch

def main():
  batch_size = 128
  matrix_queue = Type("Queue", (1, 28, 28), dtype="float32")
  mnist_type = Type("mnist_type", (1, 28, 28), dtype="float32")
  enqueue_img = EnqueueNet(matrix_queue, mnist_type)
  dequeue_img = DequeueNet(matrix_queue, mnist_type)
  enqueue_img.cuda()
  dequeue_img.cuda()
  empty_queue = Constant(matrix_queue,
                         Variable(torch.rand(1, 1, 28, 28).cuda(), requires_grad=True),
                         batch_size)
  queue_ref = {"enqueue": list_enqueue, "dequeue": list_dequeue, "empty": empty}
  neural_ref = {"enqueue": enqueue_img, "dequeue": dequeue_img, "empty": empty_queue}

  tl = trainloader(batch_size)
  train(queue_trace, tl, queue_ref, neural_ref, batch_size,
        callbacks=[plot_empty, show_ds(tl), tb_loss],
        nepochs=500)


if __name__ == "__main__":
  main()
