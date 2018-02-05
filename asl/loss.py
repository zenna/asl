from torch import nn
from multipledispatch import dispatch
import asl

def mean(xs):
  return sum(xs) / len(xs)

@dispatch(object, object)
def dist(x, y):
  return nn.MSELoss()(x, y) # TODO: Specialize this by type

@dispatch(asl.Type, asl.Type)
def dist(x, y):
  x, y = asl.modules.modules.expand_consts([x.value, y.value])
  return nn.MSELoss()(x, y) # TODO: Specialize this by type


def vec_dist(xs, ys, accumulate=mean):
  return accumulate([dist(xs[i], ys[i]) for i in range(len(xs))])
