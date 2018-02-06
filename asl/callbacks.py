import sys
import shutil
import torch
import asl
import os
import math
from asl.train import train, TrainMode
import pandas as pd

"Callbacks to be passed to optimization"
def tb_loss(i, writer, loss, **kwargs):
  "Plot loss on tensorboard"
  writer.add_scalar('data/scalar1', loss.data[0], i)


def print_stats(i, running_loss, **kwargs):
  "Print optimization statistics"
  print('[%5d] loss: %.3f' %
          (i + 1, running_loss / 2000))


def update_ret_df(opt, dfs, dffname="losses.df"):
  "Create a function which updates"
  def update_df_(i, loss, **kwargs):
    name = opt["name"]
    df = dfs[0]
    row = pd.DataFrame({'iteration': [i],
                        'runname': [name],
                        'loss': [loss]})
    dfs[0] = df.append(row)
    print(dfs)
  
  return update_df_


def save_update_df(opt, dffname="losses.df"):
  "Create a function which updates"
  name = opt["name"]
  savepath = os.path.join(opt["log_dir"], dffname)
  df = pd.DataFrame({'runname': [],
                      'iteration': [],
                      'loss': []})

  def update_df_(i, loss, **kwargs):
    nonlocal df
    # print(df)
    row = pd.DataFrame({'iteration': [i],
                        'runname': [name],
                        'loss': [loss]})
    df.append(row)

  def save_df(i, loss, **kwargs):
    print("Saving dataframe")
    nonlocal df
    df.to_pickle(savepath)

  return update_df_, save_df, df


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
        print('%s per sample (avg over %s) : %.7f' % (key, every, loss_per_sample))
        if log_tb:
          data.writer.add_scalar(key, loss_per_sample, data.i)
        running_loss = 0.0
  gen = print_loss_gen(every)
  next(gen)
  return gen

import math

def save_checkpoint(model, verbose=True, save_best=True):
  "Save model data (most recent and best)"
  best_loss = math.inf
  def save_checkpoint_innner(loss, log_dir, i, optimizer, **kwargs):
    nonlocal best_loss
    savepath = os.path.join(log_dir, "checkpoint.pt")
    if verbose:
      print("Saving checkpoint...")
    torch.save({'i': i + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                savepath)
    if save_best:
      if loss < best_loss:
        if verbose:
          print("Saving best checkpoint...")
        best_loss = loss
        bestsavepath = os.path.join(log_dir, "best_checkpoint.pt")
        torch.save({'i': i + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()},
            bestsavepath)

  return save_checkpoint_innner


def load_checkpoint(resume_path, model, optimizer, verbose=True):
  "Load data from checkpoint"
  if verbose:
    print("Loading model / optimizer from checkpoint...")
  checkpoint = torch.load(resume_path)
  optimizer.load_state_dict(checkpoint['optimizer'])
  model.load_state_dict(checkpoint['state_dict'])
  # model.eval()


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


def convergedperc(every, print_change=True, change_thres=-0.0001):
  "Converged when threshold is less than percentage? Use when optimum is zero"
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
          if running_loss == 0:
            print("Loss is zero, stopping!")
          else:
            abchange = running_loss - last_running_loss
            percchange = running_loss / last_running_loss
            print('absolute change (avg over {}) {}'.format(every, abchange))
            print('Percentage change (avg over {}) {}'.format(every, percchange))
            per_iter = percchange / every
            print('percentage_: {}, per iteration: {}'.format(percchange, per_iter))
            if per_iter < change_thres:
              print("Percentage change insufficient (< {})".format(change_thres))
              cont = False
        else:
          show_change = True
        last_running_loss = running_loss
        running_loss = 0.0

  gen = converged_gen(every)
  next(gen)
  return gen

# FIXME: This should be a cont not used as callback
def nancancel(loss, **kwargs):
  if (loss != loss):
    print("Loss is NAN: ", loss, " stopping!")
    sys.exit()


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
