from tensorboardX import SummaryWriter
from asl.callbacks import print_stats, every_n
from asl.log import getlog, reset_log
from functools import partial
import types
from collections import namedtuple
CallbackData = namedtuple('CallbackData', ['i', 'writer', 'loss', 'log'], verbose=False)

def all_epochs(i, epoch, nepochs, **kwargs):
  "Continue if we've done enough epochs"
  return epoch < nepochs

def max_iters(i, maxiters, **kwargs):
  "Continue if we've done enough epochs"
  return i < maxiters

def train(loss_gen,
          optimizer,
          callbacks=None,
          cont=partial(max_iters, maxiters=100),
          resetlog=True):
  """
  Optimization
  Args:
    loss_gen: function that returns scalar loss term to miminize
    callbacks: functions called with data every iteration, e.g for viz
    nepochs: num epochs to execute
    cont: function to determine when to stop
    resetlog: reset log data after every iteration if true
  """
  callbacks = [] if callbacks is None else callbacks
  # callbacks = callbacks + []
  writer = SummaryWriter()

  i = 0
  while cont(i=i):
    loss = loss_gen()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    cb_data = CallbackData(i, writer, loss.data[0], getlog())
    for callback in callbacks:
      if isinstance(callback, types.GeneratorType):
        callback.send(cb_data)
      else:
        callback(i=i,
                 writer=writer,
                 loss=loss.data[0],
                 log=getlog())
    i += 1
    if resetlog:
      reset_log()
  writer.close()
  print('Finished Training')
