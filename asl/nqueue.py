"Queue Data Structure trained from a reference implementation"
import itertools
from collections import deque
from type import Type, Function, FunctionType, Constant
import torchvision
import torchvision.transforms as transforms

import torch
from torch.autograd import Variable
import torch.nn as nn

from train import train
from nets import VarConvNet

import matplotlib.pyplot as plt
plt.ion()


class Enqueue(Function):
  "Push Function for Queue"

  def __init__(self, queue_type, item_type):
    super(Enqueue, self).__init__()
    self.queue_type = queue_type
    self.item_type = item_type

  def type(self):
    "Function Type"
    return FunctionType([self.queue_type, self.item_type], [self.queue_type])


class Dequeue(Function):
  "Pop Function for Queue"

  def __init__(self, queue_type, item_type):
    super(Dequeue, self).__init__()
    self.queue_type = queue_type
    self.item_type = item_type

  def type(self):
    "Function Type"
    return FunctionType([self.queue_type], [self.queue_type, self.item_type])


class EnqueueNet(Enqueue, nn.Module):
  def __init__(self, queue_type, item_type, queue_channels=1, img_channels=1):
    super(EnqueueNet, self).__init__(queue_type, item_type)
    self.module = VarConvNet(self.in_sizes(), self.out_sizes())

  def forward(self, x, y):
    return self.module.forward(x, y)


class DequeueNet(Dequeue, nn.Module):
  def __init__(self, queue_type, item_type, queue_channels=1, img_channels=1):
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


def trainloader(batch_size):
  transform = transforms.Compose(
  [transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
  return torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        shuffle=False, num_workers=1)


def draw(t):
  "Draw a tensor"
  tnp = t.data.cpu().numpy().squeeze()
  plt.imshow(tnp)
  plt.pause(0.01)


def plot_empty(i, data):
  if i % 50 == 0:
    draw(data["empty"].value)


def main():
  matrix_queue = Type("Queue", (1, 28, 28), dtype="float32")
  mnist_type = Type("mnist_type", (1, 28, 28), dtype="float32")
  enqueue_img = EnqueueNet(matrix_queue, mnist_type)
  dequeue_img = DequeueNet(matrix_queue, mnist_type)
  enqueue_img.cuda()
  dequeue_img.cuda()
  empty_queue = Constant(matrix_queue, Variable(torch.rand(1, 1, 28, 28).cuda(), requires_grad=True))
  queue_ref = {"enqueue": list_enqueue, "dequeue": list_dequeue, "empty": empty}
  neural_ref = {"enqueue": enqueue_img, "dequeue": dequeue_img, "empty": empty_queue}

  batch_size = 32
  train(queue_trace, trainloader(batch_size), queue_ref, neural_ref, batch_size,
        callbacks=[plot_empty])


if __name__ == "__main__":
  main()
