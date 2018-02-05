import torch

def onehot(i, onehot_len, batch_size):
  "Create a one hot vector from integer i"
  # Dummy input that HAS to be 2D for the scatter (you can use view(-1,1) if needed)
  y = torch.LongTensor(batch_size, 1).fill_(i)
  # One hot encoding buffer that you create out of the loop and just keep reusing
  y_onehot = torch.FloatTensor(batch_size, onehot_len)
  # In your for loop
  y_onehot.zero_()
  # FIXME: Hacked to return just one element
  return y_onehot.scatter_(1, y, 1)

def onehot2d(i, onehot_len, height, batch_size):
  "2D one hot vector"
  onehot1d = onehot(i, onehot_len, batch_size)
  unsqueezed = onehot1d.view(batch_size, 1, onehot_len, 1)
  return unsqueezed.expand(batch_size, 1, onehot_len, height)

def onehotmany(i_s, onehot_len):
  "Matrix of one hot vectors from integers i_s"
  y = torch.LongTensor(i_s).view(len(i_s), 1)
  y_onehot = torch.FloatTensor(len(i_s), onehot_len)
  y_onehot.zero_()
  y_onehot.scatter_(1, y, 1)
  return y_onehot


def onebatch(x):
  "Make a single batch out of x"
  return x.expand(1, *x.size())


def maybe_expand(cls, value, expand_one):
  sz = len(value.size())
  if sz == len(cls.typesize) and expand_one:
    return onebatch(value)
  elif (sz == len(cls.typesize) + 1) and expand_one:
    return value
  else:
    raise ValueError
