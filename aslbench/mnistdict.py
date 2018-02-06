"Stack learned from reference"
import os
from typing import List
import asl
from asl.modules.modules import ConstantNet, ModuleDict
from asl.structs.ndict import GetItem, SetItem
from torch import optim, nn
import common
from multipledispatch import dispatch
from mnist import mnist_size, Mnist, dist, refresh_mnist
from omniglot import omniglot_size, OmniGlot, dist, refresh_omniglot
from asl.structs.ndict import dict_empty, dict_get_item, dict_set_item
from stdargs import optim_sampler, arch_sampler
import torch

# dict_size = omniglot_size
# KeyType = OmniGlot
# ValueType = OmniGlot
# dataloader = asl.util.omniglotloader
# refresh_data = refresh_mnist

# dict_size = mnist_size
# KeyType = Mnist
# ValueType = Mnist
# dataloader = asl.util.mnistloader
# refresh_data = refresh_mnist

def dicttracegen(nitems):
  print("Making dict trace with {} items".format(nitems))
  def dicttrace(items, runstate, set_item, get_item, empty):
    """Example dict trace"""
    asl.log_append("empty", empty)
    adict = empty
    # keyset = [next(items) for i in range(nitems)]
    k1 = next(items)
    k2 = next(items)
    v1 = next(items)
    v2 = next(items)
    asl.log_append("{}/internal".format(runstate['mode']), v1)
    asl.log_append("{}/internal".format(runstate['mode']), v2)

    (adict,) = set_item(adict, k1, v1)
    asl.log_append("{}/internal".format(runstate['mode']), adict)

    (adict,) = set_item(adict, k2, v2)
    asl.log_append("{}/internal".format(runstate['mode']), adict)
    (v1x, ) = get_item(adict, k1)
    (v2x, ) = get_item(adict, k2)
    asl.observe(v1x, "val1", runstate)
    asl.observe(v2x, "val2", runstate)
    (adict,) = set_item(adict, k2, v1)
    asl.log_append("{}/internal".format(runstate['mode']), adict)
    (v1xx, ) = get_item(adict, k1)
    (v2xx, ) = get_item(adict, k2)
    asl.observe(v1xx, "val1_2", runstate)
    asl.observe(v2xx, "val2_2", runstate)
    return v2xx

  return dicttrace

## Dictionary training
def train_dict(opt):
  if opt["dataset"] == "omniglot":
    dict_size = mnist_size
    KeyType = OmniGlot
    ValueType = OmniGlot
    dataloader = asl.util.omniglotloader
    refresh_data = refresh_omniglot
  else:
    dict_size = mnist_size
    KeyType = Mnist
    ValueType = Mnist
    dataloader = asl.util.mnistloader
    refresh_data = refresh_mnist

  class MatrixDict(asl.Type):
    typesize = dict_size

  trace = dicttracegen(opt["nitems"])
  get_item = GetItem(MatrixDict, KeyType, ValueType, arch=opt["arch"], arch_opt=opt["arch_opt"])
  set_item = SetItem(MatrixDict, KeyType, ValueType, arch=opt["arch"], arch_opt=opt["arch_opt"])
  empty = ConstantNet(MatrixDict, init=opt["init"])
  ndict = ModuleDict({"get_item": get_item,
                      "set_item": set_item,
                      "empty": empty})
  
  class DictSketch(asl.Sketch):
    def sketch(self, items, runstate):
      """Example stack trace"""
      return trace(items, runstate, set_item=set_item, get_item=get_item, empty=empty)

  dict_sketch = DictSketch([List[Mnist]], [Mnist])

  def ref_sketch(items, runstate):
    return trace(items, runstate, set_item=dict_set_item, get_item=dict_get_item,
                 empty=dict_empty)

  asl.cuda(ndict) # CUDA that shit

  # Loss
  dataiter = dataloader(opt["batch_size"])
  loss_gen = asl.single_ref_loss(dict_sketch,
                                 ref_sketch,
                                 dataiter,
                                 refresh_data)
  return common.trainmodel(opt, ndict, loss_gen)


def dict_args(parser):
  parser.add_argument('--nitems', type=int, default=3, metavar='NI',
                      help='number of iteems in trace (default: 3)')

def dict_optspace():
  return {# "nitems": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 24, 48],
          "batch_size": [16, 32, 64],
          "arch_opt": arch_sampler,
          "optim_args": optim_sampler,
          "dataset": ["omniglot", "mnist"],
          "init": [torch.nn.init.uniform,
                   torch.nn.init.normal,
                   torch.ones_like,
                   torch.zeros_like],
          }

def runoptsgen(nsamples):
  # Delaying computation of this value because we dont know nsamples yet
  return asl.prodsample(dict_optspace(),
                        to_enum=[],
                        to_sample=["batch_size", "init", "lr", "dataset"],
                        to_sample_merge=["arch_opt", "optim_args"],
                        nsamples=nsamples)

if __name__ == "__main__":
  thisfile = os.path.abspath(__file__)
  res = common.trainloadsave(thisfile, train_dict, runoptsgen, dict_args)
