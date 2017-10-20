from typing import List
import asl
from asl.opt import opt_as_string
from asl.structs.nqueue import EnqueueNet, DequeueNet
from asl.modules.modules import ConstantNet, ModuleDict
from asl.util.misc import cuda
from asl.type import Type
from asl.sketch import Sketch
from asl.callbacks import print_loss, converged, save_checkpoint, load_checkpoint
from asl.util.data import trainloader
from asl.log import log_append
from asl.train import train
from asl.structs.nqueue import ref_queue
from torch import optim
import common


class QueueSketch(Sketch):
  def sketch(self, items, enqueue, dequeue, empty):
    """Example queue trace"""
    log_append("empty", empty)
    queue = empty
    (queue,) = enqueue(queue, next(items))
    (queue,) = enqueue(queue, next(items))
    (dequeue_queue, dequeue_item) = dequeue(queue)
    self.observe(dequeue_item)
    (dequeue_queue, dequeue_item) = dequeue(dequeue_queue)
    self.observe(dequeue_item)
    return dequeue_item


def mnist_args(parser):
  parser.add_argument('--nitems', type=int, default=3, metavar='NI',
                      help='number of iteems in trace (default: 3)')

def train_queue():
  opt = asl.opt.handle_args(mnist_args)
  opt = asl.opt.handle_hyper(opt, __file__)
  nitems = 3
  mnist_size = (1, 28, 28)

  class MatrixQueue(Type):
    size = mnist_size

  class Mnist(Type):
    size = mnist_size

  tl = trainloader(opt.batch_size)
  nqueue = ModuleDict({'enqueue': EnqueueNet(MatrixQueue, Mnist, arch=opt.arch, arch_opt=opt.arch_opt),
                       'dequeue': DequeueNet(MatrixQueue, Mnist, arch=opt.arch, arch_opt=opt.arch_opt),
                       'empty': ConstantNet(MatrixQueue)})

  queue_sketch = QueueSketch([List[Mnist]], [Mnist], nqueue, ref_queue())
  cuda(queue_sketch)
  loss_gen = asl.sketch.loss_gen_gen(queue_sketch, tl, asl.util.data.train_data)
  optimizer = optim.Adam(nqueue.parameters(), lr=opt.lr)

  asl.opt.save_opt(opt)
  if opt.resume_path is not None and opt.resume_path != '':
    load_checkpoint(opt.resume_path, nqueue, optimizer)

  train(loss_gen, optimizer, maxiters=100000,
        cont=converged(1000),
        callbacks=[print_loss(100),
                   common.plot_empty,
                   common.plot_observes,
                   save_checkpoint(1000, nqueue)],
        log_dir=opt.log_dir)


if __name__ == "__main__":
  train_queue()
