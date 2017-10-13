"Callbacks to be passed to optimization"
def tb_loss(i, writer, loss, **kwargs):
  "Plot loss on tensorboard"
  writer.add_scalar('data/scalar1', loss.data[0], i)


def print_stats(i, running_loss, **kwargs):
  "Print optimization statistics"
  print('[%5d] loss: %.3f' %
          (i + 1, running_loss / 2000))


def every_n(callback, n):
  "Higher order function that makes a callback run just once every n"
  def every_n_cb(i, **kwargs):
    if i % n == 0:
      callback(i=i, **kwargs)
  return every_n_cb


def print_loss(every):
  "Print loss per every n"
  running_loss = 0.0
  while True:
    data = yield
    i = data.i
    loss = data.loss
    running_loss += loss
    if (i + 1) % every == 0:
      print('loss per %s : %.3f' % (every, running_loss / every))
      running_loss = 0.0
