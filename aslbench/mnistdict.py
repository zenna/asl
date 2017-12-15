"Stack learned from reference"
from typing import List
import asl
from asl.modules.modules import ConstantNet, ModuleDict
from asl.structs.ndict import ref_dict
from torch import optim, nn
import common
from multipledispatch import dispatch

class DictSketch(asl.Sketch):
  def sketch(self, items, set_item, get_item, empty):
    """Example dict trace"""
    asl.log_append("empty", empty)
    adict = empty
    k1 = next(items)
    k2 = next(items)
    (adict,) = set_item(adict, k1, next(items))
    asl.log_append("{}/internal".format(self.mode.name), adict)
    (v, ) = get_item(adict, k1)
    self.observe(v)
    (adict,) = set_item(adict, k2, next(items))
    asl.log_append("{}/internal".format(self.mode.name), adict)
    (v2, ) = get_item(adict, k2)
    self.observe(v2)
    (v, ) = get_item(adict, k1)
    self.observe(v2)
    return v


def mnist_args(parser):
  parser.add_argument('--nitems', type=int, default=3, metavar='NI',
                      help='number of iteems in trace (default: 3)')

mnist_size = (1, 28, 28)

class MatrixDict(asl.Type):
  typesize = mnist_size

class Mnist(asl.Type):
  typesize = mnist_size

@dispatch(Mnist, Mnist)
def dist(x, y):
  return nn.MSELoss()(x.value, y.value)

def train_dict():
  # Get options from command line
  opt = asl.opt.handle_args(mnist_args)
  opt = asl.opt.handle_hyper(opt, __file__)

  class GetItem(asl.Function, asl.Net):
    def __init__(self="GetItem", name="GetItem", **kwargs):
      asl.Function.__init__(self, [MatrixDict, Mnist], [Mnist])
      asl.Net.__init__(self, name, **kwargs)

  class SetItem(asl.Function, asl.Net):
    def __init__(self="SetItem", name="SetItem", **kwargs):
      asl.Function.__init__(self, [MatrixDict, Mnist, Mnist], [MatrixDict])
      asl.Net.__init__(self, name, **kwargs)

  ndict = ModuleDict({'get_item': GetItem(arch=opt.arch,
                                    arch_opt=opt.arch_opt),
                       'set_item': SetItem(arch=opt.arch,
                                  arch_opt=opt.arch_opt),
                       'empty': ConstantNet(MatrixDict)})

  dict_sketch = DictSketch([List[Mnist]], [Mnist], ndict, ref_dict())
  asl.cuda(dict_sketch)

  # Loss
  mnistiter = asl.util.mnistloader(opt.batch_size)
  loss_gen = asl.sketch.loss_gen_gen(dict_sketch,
                                     mnistiter,
                                     lambda x: Mnist(asl.util.data.train_data(x)))

  # Optimization details
  optimizer = optim.Adam(ndict.parameters(), lr=opt.lr)
  asl.opt.save_opt(opt)
  if opt.resume_path is not None and opt.resume_path != '':
    asl.load_checkpoint(opt.resume_path, ndict, optimizer)

  asl.train(loss_gen, optimizer, maxiters=100000,
        cont=asl.converged(1000),
        callbacks=[asl.print_loss(100),
                   common.plot_empty,
                   common.plot_observes,
                   asl.save_checkpoint(1000, ndict)],
        log_dir=opt.log_dir)


if __name__ == "__main__":
  train_dict()
