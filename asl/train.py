"Optimization features"
from collections import namedtuple
from functools import partial
import types
from tensorboardX import SummaryWriter
from asl.log import getlog, reset_log
CallbackData = namedtuple('CallbackData', ['start_i',
                                           'i',
                                           'writer',
                                           'loss',
                                           'log',
                                           'optimizer',
                                           'log_dir',
                                           'train_mode'],
                          verbose=False)


def max_iters(i, maxiters, **kwargs):
  "Continue if we've done enough epochs"
  return i < maxiters


def apl(fungen, cb_data):
  "Apply a function or generator to data"
  if isinstance(fungen, types.GeneratorType):
    return fungen.send(cb_data)
  else:
    return fungen(**cb_data._asdict())


from enum import Enum
class TrainMode(Enum):
  PRE = 1
  RUN = 2
  POST = 3


def train(loss_gen,
          optimizer,
          writer=None,
          close_writer=False,
          pre_callbacks=None,
          callbacks=None,
          post_callbacks=None,
          maxiters=1000,
          cont=None,
          resetlog=True,
          log_dir=None,
          optimize=True,
          start_i=0):
  """
  Optimization.
  Args:
    loss_gen: function that returns scalar loss term to miminize
    opimizer: optimizer to minimize loss from loss_gen
    writer: Summary writer to log results to tensorboardX
    close_writer: Close writer after finish?
    pre_callbacks: functions/generators called before optimization
    callbacks: functions called with data every iteration, e.g for viz
    post_callbacks: functions/generators called after optimization
    maxiters: num of iterations
    cont: function to determine when to stop (overrides maxiters)
    resetlog: reset log data after every iteration if true
    log_dir: directory to store data/logs (used by callbacks)
    optimize: optimize? (compute grads/change weights)
    start_i: what index is this starting at (used by callbacks)
  """
  cont = partial(max_iters, maxiters=maxiters) if cont is None else cont
  pre_callbacks = [] if pre_callbacks is None else pre_callbacks
  post_callbacks = [] if post_callbacks is None else post_callbacks
  callbacks = [] if callbacks is None else callbacks
  writer = SummaryWriter(log_dir) if writer is None else writer

  i = 0
  cb_data = CallbackData(start_i, i, writer, None, getlog(), optimizer, log_dir,
                         TrainMode.PRE)

  # Called once before optimization
  for pre_callback in pre_callbacks:
    apl(pre_callback, cb_data)  # TODO, close/free generators?

  while apl(cont, cb_data):
    loss = loss_gen()
    if optimize:
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    cb_data = CallbackData(start_i, i, writer, loss.data[0], getlog(), optimizer,
                           log_dir, TrainMode.RUN)
    for callback in callbacks:
      apl(callback, cb_data)
    i += 1

    if resetlog:
      reset_log()

  # Post Callbacks
  cb_data = CallbackData(start_i, i, writer, None, getlog(), optimizer,
                         log_dir, TrainMode.POST)
  for callback in post_callbacks:
    apl(callback, cb_data)

  if close_writer:
    writer.close()
