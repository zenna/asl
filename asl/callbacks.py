import shutil
import torch
import asl
import os
import math
from asl.train import train, TrainMode

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


def print_loss(every, log_tb=True, key="loss"):
  "Print loss per every n"
  def print_loss_gen(every):
    running_loss = 0.0
    while True:
      data = yield
      running_loss += data.loss
      if (data.i + 1) % every == 0:
        loss_per_sample = running_loss / every
        print('%s per sample (avg over %s) : %.3f' % (key, every, loss_per_sample))
        if log_tb:
          data.writer.add_scalar(key, loss_per_sample, data.i)
        running_loss = 0.0
  gen = print_loss_gen(every)
  next(gen)
  return gen

def save_checkpoint(every, model, verbose=True):
  "Save data every every steps"
  def save_checkpoint_innner(log_dir, i, optimizer, **kwargs):
    savepath = os.path.join(log_dir, "checkpoint.pth")
    if (i + 1) % every == 0:
      if verbose:
        print("Saving...")
      torch.save({'i': i + 1,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict()},
                 savepath)
  return save_checkpoint_innner


def load_checkpoint(resume_path, model, optimizer, verbose=True):
  "Load data from checkpoint"
  if verbose:
    print("Loading...")
  torch.load(resume_path)
  checkpoint = torch.load(resume_path)
  # optimizer.load_state_dict(checkpoint['optimizer'])
  model.load_state_dict(checkpoint['state_dict'])
  model.eval()


def converged(every, print_change=True, change_thres=-0.000005):
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


def validate(test_loss_gen, maxiters=100, cont=None, pre_callbacks=None,
             callbacks=None, post_callbacks=None):
  "Validation is done using a callback"
  def validate_clos(i, optimizer, writer, **kwargs):
    print("Test Validation...")
    test_loss_cb = test_loss()
    cbs = [test_loss_cb] if callbacks is None else callbacks + [test_loss_cb]
    post_cbs = [test_loss_cb] if post_callbacks is None else post_callbacks + [test_loss_cb]

    train(test_loss_gen,
          optimizer,
          start_i = i,
          callbacks=cbs,
          pre_callbacks=pre_callbacks,
          post_callbacks=post_cbs,
          maxiters=maxiters,
          cont=cont,
          writer=writer,
          close_writer=False,
          resetlog=False,
          optimize=False)
  return validate_clos


def test_loss(log_tb=True):
  "Accumulate training loss then print/save"
  def test_loss_gen():
    running_loss = 0.0
    while True:
      data = yield
      if data.train_mode == TrainMode.RUN:
        running_loss += data.loss
      elif data.train_mode == TrainMode.POST:
        loss = running_loss / (data.i - 1)
        print('test loss per sample at %s (avg over %s) : %.3f' % (data.start_i, data.i - 1, loss))
        if log_tb:
          data.writer.add_scalar('test_loss', loss, data.start_i)
        running_loss = 0.0
  gen = test_loss_gen()
  next(gen)
  return gen
