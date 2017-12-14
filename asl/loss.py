from torch import nn
from multipledispatch import dispatch

def mean(xs):
  return sum(xs) / len(xs)

@dispatch(object, object)
def dist(x, y):
  return nn.MSELoss()(x, y) # TODO: Specialize this by type

def vec_dist(xs, ys, accumulate=mean):
  return accumulate([dist(xs[i], ys[i]) for i in range(len(xs))])
