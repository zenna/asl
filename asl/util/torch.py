import torch
from torch.autograd import Variable
from asl.util.misc import cuda


def onehot(i, onehot_len, batch_size):
  "Create a one hot vector from integer i"
  # Dummy input that HAS to be 2D for the scatter (you can use view(-1,1) if needed)
  y = torch.LongTensor(batch_size, 1).fill_(i)
  # One hot encoding buffer that you create out of the loop and just keep reusing
  y_onehot = torch.FloatTensor(batch_size, onehot_len)
  # In your for loop
  y_onehot.zero_()
  return Variable(cuda(y_onehot.scatter_(1, y, 1)), requires_grad=False)


def onehotmany(i_s, onehot_len):
  "Createa a matrix of one hot vectors from integers i_s"
  y = torch.LongTensor(i_s).view(len(i_s), 1)
  y_onehot = torch.FloatTensor(len(i_s), onehot_len)
  y_onehot.zero_()
  y_onehot.scatter_(1, y, 1)
  return y_onehot
