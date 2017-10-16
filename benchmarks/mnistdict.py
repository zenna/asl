from typing import List
import asl
import asl.opt
from asl.structs.ndict import SetItemNet, GetItemNet
from asl.modules.modules import ConstantNet, ModuleDict
from asl.util.misc import cuda
from asl.type import Type
from asl.sketch import Sketch
from asl.callbacks import print_loss, converged, save_checkpoint, load_checkpoint
from asl.util.data import trainloader
from asl.log import log_append
from asl.train import train
from asl.structs.ndict import ref_dict
from torch import optim
import common


class DictSketch(Sketch):
  def sketch(self, items, set_item, get_item, empty):
    """Example dict trace"""
    log_append("empty", empty)
    adict = empty
    k1 = next(items)
    k2 = next(items)
    (adict,) = set_item(adict, k1, next(items))
    (v, ) = get_item(adict, k1)
    self.observe(v)
    (adict,) = set_item(adict, k2, next(items))
    (v2, ) = get_item(adict, k2)
    self.observe(v2)
    (v, ) = get_item(adict, k1)
    self.observe(v2)
    return v


def mnist_args(parser):
  parser.add_argument('--opt.nitems', type=int, default=3, metavar='NI',
                      help='number of iteems in trace (default: 3)')

def train_dict():
  opt = asl.opt.handle_args(mnist_args)
  opt = asl.opt.handle_hyper(opt, __file__)
  opt.nitems = 3
  mnist_size = (1, 28, 28)

  class MatrixDict(Type):
    size = mnist_size

  class Mnist(Type):
    size = mnist_size

  tl = trainloader(opt.batch_size)
  ndict = ModuleDict({'set_item': SetItemNet(MatrixDict, Mnist, Mnist, template=opt.template, template_opt=opt.template_opt),
                       'get_item': GetItemNet(MatrixDict, Mnist, Mnist, template=opt.template, template_opt=opt.template_opt),
                       'empty': ConstantNet(MatrixDict)})

  dict_sketch = DictSketch([List[Mnist]], [Mnist], ndict, ref_dict())
  cuda(dict_sketch)
  loss_gen = asl.sketch.loss_gen_gen(dict_sketch, tl, asl.util.data.train_data)
  optimizer = optim.Adam(ndict.parameters(), lr=opt.lr)

  asl.opt.save_opt(opt)
  if opt.resume_path is not None and opt.resume_path != '':
    load_checkpoint(opt.resume_path, ndict, optimizer)

  train(loss_gen, optimizer, maxiters=100000,
        cont=converged(1000),
        callbacks=[print_loss(100),
                   common.plot_empty,
                   common.plot_observes,
                   save_checkpoint(1000, ndict)],
        log_dir=opt.log_dir)


if __name__ == "__main__":
  train_dict()
