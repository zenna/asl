"Stack learned from reference"
import os
from typing import List
import asl
from asl.modules.modules import ConstantNet, ModuleDict
from asl.structs.ndict import ref_dict
from torch import optim, nn
import common
from multipledispatch import dispatch
from mnist import mnist_size, Mnist, dist, refresh_mnist
from asl.structs.ndict import dict_empty, dict_get_item, dict_set_item
from stdargs import optim_sampler, arch_sampler

# class DictSketch(asl.Sketch):

def dicttracegen(nitems, nrounds):
  print("Making dict trace with {} items and {} rounds".format(nitems, nrounds))
  def dicttrace(items, runstate, set_item, get_item, empty):
    """Example dict trace"""
    asl.log_append("empty", empty)
    adict = empty
    # keyset = [next(items) for i in range(nitems)]
    k1 = next(items)
    k2 = next(items)
    (adict,) = set_item(adict, k1, next(items))
    asl.log_append("{}/internal".format(runstate['mode']), adict)
    (v, ) = get_item(adict, k1)
    asl.observe(v, "val1", runstate)
    (adict,) = set_item(adict, k2, next(items))
    asl.log_append("{}/internal".format(runstate['mode']), adict)
    (v2, ) = get_item(adict, k2)
    asl.observe(v2, "val2a", runstate)
    (v, ) = get_item(adict, k1)
    asl.observe(v2, "val2a", runstate)
    return v

  return dicttrace

class MatrixDict(asl.Type):
  typesize = mnist_size

class GetItem(asl.Function, asl.Net):
  def __init__(self="GetItem", name="GetItem", **kwargs):
    asl.Function.__init__(self, [MatrixDict, Mnist], [Mnist])
    asl.Net.__init__(self, name, **kwargs)

class SetItem(asl.Function, asl.Net):
  def __init__(self="SetItem", name="SetItem", **kwargs):
    asl.Function.__init__(self, [MatrixDict, Mnist, Mnist], [MatrixDict])
    asl.Net.__init__(self, name, **kwargs)

## Dictionary training
def train_dict(opt):
  trace = dicttracegen(opt["nitems"], opt["nrounds"])
  get_item = GetItem(arch=opt.arch, arch_opt=opt.arch_opt)
  set_item = SetItem(arch=opt.arch, arch_opt=opt.arch_opt)
  empty = ConstantNet(MatrixDict)
  ndict = ModuleDict({"get_item": get_item,
                      "set_item": set_item,
                      "empty": empty})
  dict_sketch = DictSketch([List[Mnist]], [Mnist], ndict, ref_dict())

  def ref_sketch(items, runstate):
    return trace(items, runstate, set_item=dict_set_item, get_item=dict_get_item,
                 empty=dict_empty)

  asl.cuda(ndict) # CUDA that shit

  # Loss
  mnistiter = asl.util.mnistloader(opt.batch_size)
  loss_gen = asl.sketch.single_ref_loss(dict_sketch,
                                        ref_sketch,
                                        mnistiter,
                                        refresh_mnist)
  return common.trainmodel(opt, ndict, loss_gen)


def dict_args(parser):
  parser.add_argument('--nitems', type=int, default=3, metavar='NI',
                      help='number of iteems in trace (default: 3)')
  parser.add_argument('--nrounds', type=int, default=1, metavar='NR',
                      help='number of rounds in trace')


def dict_optspace():
  return {"nrounds": [1, 2],
          "nitems": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 24, 48],
          "batch_size": [8, 16, 32, 64, 128],
          "arch_opt": arch_sampler,
          "optim_args": optim_sampler}

def runoptsgen(nsamples):
  # Delaying computation of this value because we dont know nsamples yet
  return asl.prodsample(dict_optspace(),
                        to_enum=["nitems"],
                        to_sample=["batch_size", "lr", "nrounds"],
                        to_sample_merge=["arch_opt"],
                        nsamples=nsamples)

if __name__ == "__main__":
  thisfile = os.path.abspath(__file__)
  res = common.trainloadsave(thisfile, train_dict, runoptsgen, dict_args)
