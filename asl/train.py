from tensorboardX import SummaryWriter
from asl.callbacks import print_stats, every_n
from asl.log import getlog, reset_log
from functools import partial

def all_epochs(i, epoch, nepochs, **kwargs):
  "Continue if we've done enough epochs"
  return epoch < nepochs

def max_iters(i, maxiters, **kwargs):
  "Continue if we've done enough epochs"
  return i < maxiters

def train(loss_gen,
          optimizer,
          callbacks=None,
          cont=partial(max_iters, maxiters=10),
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
  callbacks = callbacks + [every_n(print_stats, 100)]
  writer = SummaryWriter()

  i = 0
  running_loss = 0.0
  while cont(i=i):
    loss = loss_gen()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    for callback in callbacks:
      callback(i=i,
               writer=writer,
               loss=loss,
               running_loss=running_loss,
               log=getlog())
    running_loss += loss.data[0]
    i += 1
    if resetlog:
      reset_log()
  writer.close()
  print('Finished Training')
