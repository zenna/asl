"Common Types"
from asl.type import Type
import torch
from torch.autograd import Variable
from asl.util.misc import cuda
from asl.util.generators import infinite_samples

def vec_stack(string_len):
  class VecStack(Type):
    "Stack represented as a vector"
    size = (string_len,)
  return VecStack

def matrix_stack(nchannels, width, height):
  class MatrixStack(Type):
    "Stack represented as a vector"
    size = (nchannels, width, height)
  return MatrixStack


def vec_queue(string_len):
  class VecQueue(Type):
    "Stack represented as a vector"
    size = (string_len,)
  return VecQueue

def bern_seq(string_len):
  class BernSeq(Type):
    "Bernoulli Sequence"
    size = (string_len,)
    def sample(*shape):
      return Variable(cuda(torch.bernoulli(torch.ones(*shape).fill_(0.5))))

    def iter(batch_size):
      return infinite_samples(BernSeq.sample, batch_size, (string_len,), True)
  return BernSeq
