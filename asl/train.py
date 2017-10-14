"Optimization features"
from collections import namedtuple
from functools import partial
import types
from tensorboardX import SummaryWriter
from asl.log import getlog, reset_log
CallbackData = namedtuple('CallbackData', ['i',
                                           'writer',
                                           'loss',
                                           'log',
                                           'optimizer',
                                           'log_dir'],
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


def train(loss_gen,
          optimizer,
          callbacks=None,
          pre_cbs=None,
          maxiters=1000,
          cont=None,
          resetlog=True,
          log_dir=None):
  """
  Optimization.
  Args:
    loss_gen: function that returns scalar loss term to miminize
    callbacks: functions called with data every iteration, e.g for viz
    maxiters: num of iterations
    cont: function to determine when to stop (overrides maxiters)
    resetlog: reset log data after every iteration if true
  """
  cont = partial(max_iters, maxiters=maxiters) if cont is None else cont
  pre_cbs = [] if pre_cbs is None else pre_cbs
  callbacks = [] if callbacks is None else callbacks
  writer = SummaryWriter(log_dir)

  i = 0
  cb_data = CallbackData(i, writer, None, getlog(), optimizer, log_dir)

  # Called once before optimization
  for pre_cb in pre_cbs:
    apl(pre_cb, cb_data)  # TODO, close/free generators?

  while apl(cont, cb_data):
    loss = loss_gen()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    cb_data = CallbackData(i, writer, loss.data[0], getlog(), optimizer, log_dir)
    for callback in callbacks:
      apl(callback, cb_data)
    i += 1
    if resetlog:
      reset_log()
  writer.close()
  print('Finished Training')
