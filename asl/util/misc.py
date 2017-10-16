"Miscellaneous Utilities"
from functools import reduce
import operator
import os
from distutils.util import strtobool
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
plt.ion()

def imap(f, itr):
  "Return an iterator which applies f to output of iterator"
  while True:
    yield f(next(itr))


def use_gpu(default=True):
  "Check environment variable USE_GPU"
  if "USE_GPU" in os.environ:
    return strtobool(os.environ["USE_GPU"])
  else:
    return default


def cuda(tensor, use_cuda=None):
  "Put tensor on GPU (maybe)"
  use_cuda = use_gpu() if use_cuda is None else use_cuda
  if use_cuda:
    return tensor.cuda()
  else:
    return tensor


def as_img(t):
  return t.data.cpu().numpy().squeeze()


def draw(t):
  "Draw a tensor"
  tnp = as_img(t)
  plt.imshow(tnp)
  plt.pause(0.01)


def identity(x):
  return x


def iterget(dataiter, n, transform=identity):
  return [Variable(cuda(transform(next(dataiter)))) for i in range(n)]


def take(iter, n):
  "Take n items for iterator"
  return [next(iter) for i in range(n)]


def is_tensor_var(tensor):
  "Is tensor either a tensor or variable?"
  return isinstance(tensor, torch.autograd.Variable) or torch.is_tensor(tensor)


def mul_product(xs):
  "x1 * x2 ... xn for all xi in xs"
  return reduce(operator.mul, xs, 1)
