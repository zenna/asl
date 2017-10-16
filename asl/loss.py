from torch import nn

def mean(xs):
  return sum(xs) / len(xs)


def dist(x, y):
  return nn.MSELoss()(x, y) # TODO: Specialize this by type


def vec_dist(xs, ys, dist=nn.MSELoss(), accumulate=mean):
    return accumulate([dist(xs[i], ys[i]) for i in range(len(xs))])
