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


def print_loss(every, log_tb=True):
  "Print loss per every n"
  def print_loss_gen(every):
    running_loss = 0.0
    while True:
      data = yield
      running_loss += data.loss
      if (data.i + 1) % every == 0:
        loss_per_sample = running_loss / every
        print('loss per sample (avg over %s) : %.3f' % (every, loss_per_sample))
        if log_tb:
          data.writer.add_scalar('loss', loss_per_sample, data.i)
        running_loss = 0.0
  gen = print_loss_gen(every)
  next(gen)
  return gen


def converged(every, print_change=True, change_thres=-0.00005):
  "Has the optimization converged?"
  def converged_gen(every):
    running_loss = 0.0
    last_running_loss = 0.0
    show_change = False
    cont = True
    while True:
      data = yield cont
      if data.loss is None:
        continue
      running_loss += data.loss
      if (data.i + 1) % every == 0:
        if show_change:
          change = (running_loss - last_running_loss)
          print('absolute change (avg over {}) {}'.format(every, change))
          if last_running_loss != 0:
            relchange = change / last_running_loss
            per_iter = relchange / every
            print('relative_change: {}, per iteration: {}'.format(relchange,
                                                                  per_iter))
            if per_iter > change_thres:
              print("Relative change insufficeint, stopping!")
              cont = False
        else:
          show_change = True
        last_running_loss = running_loss
        running_loss = 0.0

  gen = converged_gen(every)
  next(gen)
  return gen
