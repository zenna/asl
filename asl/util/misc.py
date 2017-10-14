"Miscellaneous Utilities"
import os
from distutils.util import strtobool
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
plt.ion()


def use_gpu(def_use_gpu=True):
  if "USE_GPU" in os.environ:
    return strtobool(os.environ["USE_GPU"])
  else:
    return def_use_gpu


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


def train_data(data):
  return data[0]


def identity(x):
  return x


def iterget(dataiter, n, transform=identity):
  return [Variable(cuda(transform(next(dataiter)))) for i in range(n)]


def is_tensor_var(tensor):
  "Is tensor either a tensor or variable?"
  return isinstance(tensor, torch.autograd.Variable) or torch.is_tensor(tensor)
