import torch
import torch.nn as nn
from asl.archs import *
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

def f(x, plus):
  y = plus(x, x)
  return y

a_size = (10,)
batch_size = 128

## Functions
## =========

# +': x::A, y::A -> z::A
plus_a = asl.archs.MLPNet(in_sizes = [a_size, a_size], out_sizes = [a_size], batch_norm=False)

# isin: x::R, a::A -> Bool
bool_size = (1, ) # Boolean one hot
x_size = (1,) # Shape of input
isin = asl.archs.MLPNet(in_sizes = [x_size, a_size], out_sizes = [bool_size], batch_norm=False)
# isin = nn.Sequential(isin_, nn.Sigmoid())

# construct: lb, ub -> A
construct = asl.archs.MLPNet(in_sizes = [x_size, x_size], out_sizes = [a_size], batch_norm=False)

# TODO: Make network
alice_x_ = torch.autograd.Variable(torch.rand(batch_size, *x_size), requires_grad=True)
alice_y_ = torch.autograd.Variable(torch.rand(batch_size, *x_size), requires_grad=True)

def alice_x():
    return alice_x_

def alice_y():
    return alice_y_

def x_loss(x, a, b):
    a = Variable(torch.Tensor(x.size()).fill_(a))
    b = Variable(torch.Tensor(x.size()).fill_(b))
    z = torch.max(torch.zeros_like(x), torch.max(x - b, a-x))
    return torch.sigmoid(z)

def plus_loss(x, y):
    t1 = ((y - x) / 2)
    t2 = ((x - y) / 2)
    z = torch.sqrt(t1 * t1 + t2 * t2)
    return torch.sigmoid(z)

def iff(a, b):
    "a iff b in one hot encoding"
    # FIXME: use cross entropy
    # return nn.functional.cross_entropy(a, b)
    diff = a - b
    diffsqr = diff * diff
    return diffsqr.mean()
    # return nn.functional.mse_loss(a, b)

min_params = list(plus_a.parameters()) + \
             list(isin.parameters()) + \
             list(construct.parameters())

max_params = [alice_x_, alice_y_]

# Two optimizers for parameters to min / max
# Use ADAM?
min_optimizer = optim.SGD(min_params, lr=0.001, momentum=0.9)
max_optimizer = optim.SGD(max_params, lr=0.001, momentum=0.9)

## Training
i = 0
while True:
    min_optimizer.zero_grad()
    max_optimizer.zero_grad()

    (x_a, ) = construct(Variable(torch.Tensor([[2.0]])),
                        Variable(torch.Tensor([[3.0]])))
    
    alice_x__ = alice_x()
    (x_isin, ) = isin(alice_x__, x_a)
    x_isin = torch.sigmoid(x_isin)
    x_loss_ = x_loss(alice_x__, 2.0, 3.0)

    l1 = iff(x_isin, x_loss_)

    (y_a, ) = f(x_a, plus_a)
    alice_y__ = alice_y()
    (y_is_in, ) = isin(alice_y__, y_a)
    y_is_in = torch.sigmoid(y_is_in)
    y_loss_ = plus_loss(alice_x__, alice_y__)

    l2 = iff(y_is_in, y_loss_)

    l = l1 + l2
    # print("loss is", l)

    l.backward(retain_graph=True)
    min_optimizer.step()
    minus_l = -l
    minus_l.backward()
    max_optimizer.step()
    i = i + 1

    if i % 100 == 0:
        query = torch.autograd.Variable(torch.Tensor([[1.0]]))
        (x_isin, ) = isin(query, x_a)
        x_isin = torch.sigmoid(x_isin)
        query2 = torch.autograd.Variable(torch.Tensor([[2.5]]))
        (x_isin2, ) = isin(query2, x_a)
        x_isin2 = torch.sigmoid(x_isin2)
        print("query", x_isin, x_isin2)
        x_loss_1 = x_loss(query, 2.0, 3.0)
        import pdb; pdb.set_trace()
        x_loss_2 = x_loss(query2, 2.0, 3.0)
        print("xlosses", x_loss_1, x_loss_2)
        print("biimpls", iff(x_isin, x_loss_1), iff(x_isin2, x_loss_2))


        