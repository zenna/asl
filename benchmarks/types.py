"Common Types"
from asl.type import Type
import torch
from torch.autograd import Variable
from asl.util.misc import cuda
from asl.util.generators import infinite_samples

def vec_stack(string_len):
  class VecStack(Type):
    "Stack represented as a vector"
    size = (string_len, 1)
  return VecStack


def vec_queue(string_len):
  class VecQueue(Type):
    "Stack represented as a vector"
    size = (string_len, 1)
  return VecQueue

def bern_seq(string_len):
  class BernSeq(Type):
    "Bernoulli Sequence"
    size = (string_len, 1)
    def sample(*shape):
      return Variable(cuda(torch.bernoulli(torch.ones(*shape).fill_(0.5))))

    def iter(batch_size):
      return infinite_samples(BernSeq.sample, batch_size, (string_len, 1), True)
  return BernSeq
