"Queue Data Structure trained from a reference implementation"
import itertools
from collections import deque
from type import Type, Function, FunctionType, Constant

import torch
from torch.autograd import Variable
import torch.nn as nn

from train import train
from nets import VarConvNet

from util import draw, trainloader


class SetItem(Function):
  "Add k:i to dict"

  def __init__(self, dict_type, key_type, item_type):
    super(SetItem, self).__init__()
    self.dict_type = dict_type
    self.key_type = key_type
    self.item_type = item_type

  def type(self):
    "Function Type"
    return FunctionType([self.dict_type, self.key_type, self.item_type],
                        [self.dict_type])


class GetItem(Function):
  "Get value with key from dict"

  def __init__(self, dict_type, key_type, item_type):
    super(GetItem, self).__init__()
    self.dict_type = dict_type
    self.key_type = key_type
    self.item_type = item_type

  def type(self):
    "Function Type"
    return FunctionType([self.dict_type, self.key_type],
                        [self.item_type])


class SetItemNet(SetItem, nn.Module):
  def __init__(self, dict_type, key_type, item_type):
    super(SetItemNet, self).__init__(dict_type, key_type, item_type)
    self.module = VarConvNet(self.in_sizes(), self.out_sizes())

  def forward(self, d, k, v):
    return self.module.forward(d, k, v)


class GetItemNet(GetItem, nn.Module):
  def __init__(self, dict_type, key_type, item_type):
    super(GetItemNet, self).__init__(dict_type, key_type, item_type)
    self.module = VarConvNet(self.in_sizes(), self.out_sizes())

  def forward(self, d, k):
    return self.module.forward(d, k)


def dict_set_item(dict, key, item):
  dict = dict.copy()
  dict[key] = item
  return (dict, )


def dict_get_item(dict, key):
  item = dict[key]
  return (item, )


def empty_dict():
  return {}


def dict_trace(items, get_item, set_item, empty):
  """Example dict trace"""
  items = [Variable(data[0].cuda()) for data in list(itertools.islice(items, 3))]
  # if expand_empty:
  empty = empty()
  observes = []
  dict = empty
  # print("mean", items[0].mean())
  (dict,) = set_item(dict, items[0], items[1])
  (item,) = get_item(dict, items[0])
  observes.append(item)
  return observes


def plot_empty(i, data):
  if i % 50 == 0:
    draw(data["empty"].value)


def main():
  matrix_dict = Type("Dict", (1, 28, 28), dtype="float32")
  mnist_type = Type("mnist_type", (1, 28, 28), dtype="float32")
  set_item_img = SetItemNet(matrix_dict, mnist_type, mnist_type)
  get_item_img = GetItemNet(matrix_dict, mnist_type, mnist_type)
  set_item_img.cuda()
  get_item_img.cuda()
  empty_ndict = Constant(matrix_dict, Variable(torch.rand(1, 1, 28, 28).cuda(), requires_grad=True))
  dict_ref = {"set_item": dict_set_item, "get_item": dict_get_item, "empty": empty_dict}
  neural_ref = {"set_item": set_item_img, "get_item": get_item_img, "empty": empty_ndict}

  batch_size = 32
  train(dict_trace, trainloader(batch_size), dict_ref, neural_ref, batch_size,
        callbacks=[plot_empty])


if __name__ == "__main__":
  main()
