"Set learned from reference"
import torch
import random
import os
from typing import List
import asl
from asl.modules.modules import ConstantNet, ModuleDict
from asl.structs.nset import *
from torch import optim, nn
from torch.autograd import Variable
import common
from stdargs import optim_sampler, arch_sampler
from mnist import mnist_size, Mnist, dist, refresh_mnist
from asl.callbacks import every_n
from multipledispatch import dispatch
from asl.loss import mean

def tracegen(nitems, nrounds):
  print("Making set trace with {} items and {} rounds".format(nitems, nrounds))
  def trace(items, r, runstate, add, card, empty):
    """Example set trace"""
    # import pdb; pdb.set_trace()
    asl.log_append("empty", empty)
    aset = empty
    (set_card, ) = card(aset)
    asl.observe(bridge(set_card), "card1", runstate)
    i1 = next(items)
    i2 = next(items)
    (aset, ) = add(aset, i1)
    asl.log_append("{}/internal".format(runstate['mode']), aset)
    (aset, ) = add(aset, i1)
    asl.observe(bridge(set_card), "card2", runstate)      
    asl.log_append("{}/internal".format(runstate['mode']), aset)
    (aset, ) = add(aset, i2)
    asl.log_append("{}/internal".format(runstate['mode']), aset)
    (set_card, ) = card(aset)
    asl.observe(bridge(set_card), "card3", runstate)      
    return set_card
  
  return trace


def tracegen2(nitems, nrounds):
  print("Making set trace with {} items and {} rounds".format(nitems, nrounds))
  def trace(items, r, runstate, add, card, empty):
    """Example set trace"""
    asl.log_append("empty", empty)
    aset = empty
    (set_card, ) = card(aset)
    asl.observe(bridge(set_card), "card1", runstate)
    hand = [next(items) for i in range(5)]
    subhand = [r.choice(hand) for i in range(5)]
    for i, item in enumerate(subhand):
      (aset, ) = add(aset, item)
      asl.log_append("{}/internal".format(runstate['mode']), aset)
      (set_card, ) = card(aset)
      asl.observe(bridge(set_card), "card.{}".format(i), runstate)
    return set_card
  
  return trace

ItemType = Mnist

## Data structures and functions
class Bool(asl.Type):
  type = mnist_size

class MatrixSet(asl.Type):
  typesize = mnist_size

class MatrixInteger(asl.Type):
  typesize = mnist_size

@dispatch(Mnist, Mnist)
def dist(x, y):
  return nn.MSELoss()(x.value, y.value)

class PyBoolToNBool(asl.Function, asl.Net):
  def __init__(self, SetType, name="Intersection", **kwargs):
    asl.Function.__init__(self, [SetType, SetType], [SetType])
    asl.Net.__init__(self, name, **kwargs)


@dispatch(MatrixInteger)
def bridge(mi):
  return mi

## Training
def train_set(opt):
  trace = tracegen2(opt["nitems"], opt["nrounds"])
  add = Add(MatrixSet, Mnist, arch=opt["arch"], arch_opt=opt["arch_opt"])
  card = Card(MatrixSet, MatrixInteger, arch=opt["arch"], arch_opt=opt["arch_opt"])
  empty = ConstantNet(MatrixSet,
                      requires_grad=opt["learn_constants"],
                      init=opt["init"])

  def ref_sketch(items, seed, runstate):
    return trace(items, seed, runstate, add=py_add, card=py_card, empty=py_empty_set)

  class SetSketch(asl.Sketch):
    def sketch(self, items, seed, runstate):
      """Example set trace"""
      return trace(items, seed, runstate, add=add, card=card, empty=empty)

  set_sketch = SetSketch([List[Mnist]], [Mnist])
  nset = ModuleDict({"add": add,
                     "card": card,
                     "empty": empty,
                     "set_sketch": set_sketch})

  # Cuda that shit
  asl.cuda(nset, opt["nocuda"])

  @dispatch(int)
  def bridge(pyint):
    width = mnist_size[1]
    height = mnist_size[2]
    res = asl.util.onehot2d(pyint, width, height, opt["batch_size"])
    res = Variable(res, requires_grad=False)
    return MatrixInteger(asl.cuda(res, opt["nocuda"]))

  # Loss (horrible horrible hacking)
  mnistiter = asl.util.mnistloader(opt["batch_size"])

  # Make two random number objects which should stay in concert
  model_random, ref_random = random.Random(0), random.Random(0)
  model_inputs = [mnistiter, model_random]
  ref_inputs = [mnistiter, ref_random]
  
  def refresh_inputs(x):
    mnist_dl, randomthing = x
    a = asl.refresh_iter(mnist_dl, lambda x: Mnist(asl.util.image_data(x)))
    return [a, random.Random(0)]

  loss_gen = asl.single_ref_loss_diff_inp(set_sketch,
                                          ref_sketch,
                                          model_inputs,
                                          ref_inputs,
                                          refresh_inputs,
                                          accum=opt["accum"])
  if opt["learn_constants"]:
    parameters = nset.parameters()
  else:
    parameters = torch.nn.ModuleList([add, card]).parameters()

  return common.trainmodel(opt, nset, loss_gen, parameters)

## Samplers

def set_args(parser):
  parser.add_argument('--nitems', type=int, default=3, metavar='NI',
                      help='number of iteems in trace (default: 3)')
  parser.add_argument('--nrounds', type=int, default=1, metavar='NR',
                      help='number of rounds in trace')

def set_optspace():
  return {"nrounds": [1, 2],
          "nitems": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20],
          "batch_size": [32, 64, 128, 256, 512],
          "learn_constants": [True, False],
          "accum": [mean],
          "init": [torch.nn.init.uniform,
                   torch.nn.init.normal,
                   torch.nn.init.xavier_normal,
                   torch.nn.init.xavier_uniform,
                   torch.nn.init.kaiming_normal,
                   torch.nn.init.kaiming_uniform,
                   torch.ones_like,
                   torch.zeros_like],
          "arch_opt": arch_sampler,
          "optim_args": optim_sampler}

def runoptsgen(nsamples):
  # Delaying computation of this value because we dont know nsamples yet
  return asl.prodsample(set_optspace(),
                        to_enum=[],
                        to_sample=["init", "batch_size", "lr", "accum", "learn_constants"],
                        to_sample_merge=["arch_opt", "optim_args"],
                        nsamples=nsamples)

if __name__ == "__main__":
  thisfile = os.path.abspath(__file__)
  res = common.trainloadsave(thisfile, train_set, runoptsgen, set_args)