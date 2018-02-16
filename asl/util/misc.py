"Miscellaneous Utilities"
from functools import reduce
import operator
import os
from distutils.util import strtobool
import torch
from torch.autograd import Variable
import torch.nn.functional as F

# import matplotlib.pyplot as plt
# plt.ion()
def mergedict(d1, d2):
  "Merge opts, opt1 takes precedence"
  opt = copy(d1)
  opt.update(d2)
  return opt

def invert(d):
  "Assume d is injective"
  return dict(zip(d.values(), d.keys()))

STRINGTOF = {"elu": F.elu,
             "relu": F.relu}
FTOSTRING = invert(STRINGTOF)

def repl(tpl, index, newval):
  """functional update: tpl[index] = newval
  In [3]: repl((1,2,3), 1, 3)
  Out[3]: (1, 3, 3)
  """
  tpllist = [*tpl]
  tpllist[index] = newval
  return tuple(tpllist)


def addbatchdim(size, batch_size=-1):
  return (batch_size,) + size


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


def cuda(tensor, nocuda=False):
  "Put tensor on GPU (maybe)"
  # nocuda = use_gpu() if nocuda is None else nocuda
  if not nocuda and torch.cuda.is_available():
    return tensor.cuda()
  else:
    return tensor


def as_img(t):
  return t.data.cpu().numpy().squeeze()


# def draw(t):
#   "Draw a tensor"
#   tnp = as_img(t)
#   plt.imshow(tnp)
#   plt.pause(0.01)
#

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
