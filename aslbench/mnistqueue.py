"Queue learned from reference"
from typing import List
import asl
from asl.modules.modules import ConstantNet, ModuleDict
from asl.structs.nqueue import ref_queue
from torch import optim, nn
import common
from multipledispatch import dispatch

class QueueSketch(asl.Sketch):
  def sketch(self, items, enqueue, dequeue, empty):
    """Example queue trace"""
    asl.log_append("empty", empty)
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

mnist_size = (1, 28, 28)

class MatrixQueue(asl.Type):
  typesize = mnist_size

class Mnist(asl.Type):
  typesize = mnist_size

@dispatch(Mnist, Mnist)
def dist(x, y):
  return nn.MSELoss()(x.value, y.value)

def train_queue():
  # Get options from command line
  opt = asl.opt.handle_args(mnist_args)
  opt = asl.opt.handle_hyper(opt, __file__)

  class Enqueue(asl.Function, asl.Net):
    def __init__(self="Enqueue", name="Enqueue", **kwargs):
      asl.Function.__init__(self, [MatrixQueue, Mnist], [MatrixQueue])
      asl.Net.__init__(self, name, **kwargs)

  class Dequeue(asl.Function, asl.Net):
    def __init__(self="Dequeue", name="Dequeue", **kwargs):
      asl.Function.__init__(self, [MatrixQueue], [MatrixQueue, Mnist])
      asl.Net.__init__(self, name, **kwargs)

  nqueue = ModuleDict({'enqueue': Enqueue(arch=opt.arch,
                                          arch_opt=opt.arch_opt),
                       'dequeue': Dequeue(arch=opt.arch,
                                          arch_opt=opt.arch_opt),
                       'empty': ConstantNet(MatrixQueue)})

  queue_sketch = QueueSketch([List[Mnist]], [Mnist], nqueue, ref_queue())
  asl.cuda(queue_sketch)

  # Loss
  mnistiter = asl.util.mnistloader(opt.batch_size)
  loss_gen = asl.sketch.loss_gen_gen(queue_sketch,
                                     mnistiter,
                                     lambda x: Mnist(asl.util.data.train_data(x)))

  # Optimization details
  optimizer = optim.Adam(nqueue.parameters(), lr=opt.lr)
  asl.opt.save_opt(opt)
  if opt.resume_path is not None and opt.resume_path != '':
    asl.load_checkpoint(opt.resume_path, nqueue, optimizer)

  asl.train(loss_gen, optimizer, maxiters=100000,
        cont=asl.converged(1000),
        callbacks=[asl.print_loss(100),
                   common.plot_empty,
                   common.plot_observes,
                   asl.save_checkpoint(1000, nqueue)],
        log_dir=opt.log_dir)


if __name__ == "__main__":
  train_queue()
