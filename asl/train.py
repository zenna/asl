from tensorboardX import SummaryWriter
from asl.callbacks import print_stats, every_n
from asl.log import getlog, reset_log

def train(loss_gen,
          optimizer,
          callbacks=None,
          nepochs=10,
          resetlog=True):
  callbacks = [] if callbacks is None else callbacks
  callbacks = callbacks + [every_n(print_stats, 100)]
  writer = SummaryWriter()

  i = 0
  for epoch in range(nepochs):
    running_loss = 0.0
    while True:
      loss = loss_gen()
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      for callback in callbacks:
        callback(i=i,
                 writer=writer,
                 loss=loss,
                 epoch=epoch,
                 running_loss=running_loss,
                 log=getlog())
      running_loss += loss.data[0]
      i += 1
      if resetlog:
        reset_log()
  writer.close()
  print('Finished Training')
