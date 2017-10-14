"Optimization features"
from collections import namedtuple
from functools import partial
import types
from tensorboardX import SummaryWriter
from asl.log import getlog, reset_log
CallbackData = namedtuple('CallbackData', ['i', 'writer', 'loss', 'log'],
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
          maxiters=1000,
          cont=None,
          resetlog=True):
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
  callbacks = [] if callbacks is None else callbacks
  writer = SummaryWriter()

  i = 0
  cb_data = CallbackData(i, writer, None, getlog())
  while apl(cont, cb_data):
    loss = loss_gen()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    cb_data = CallbackData(i, writer, loss.data[0], getlog())
    for callback in callbacks:
      apl(callback, cb_data)
    i += 1
    if resetlog:
      reset_log()
  writer.close()
  print('Finished Training')
