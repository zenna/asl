# from mnistset import record_data

import asl
import pandas as pd
import os
import stdargs
import numpy as np
# import matplotlib.pyplot as plt

def record_data(train_f, path, opt_update=None):
  opt_update = {} if opt_update is None else opt_update
  optfile = os.path.join(path, "opt.pkl")
  best_checkpoint = os.path.join(path, "best_checkpoint.pt")
  opt = asl.load_opt(optfile)
  opt["resume_path"] = best_checkpoint
  opt["log_dir"] = "/tmp/.{}".format(opt["name"])
  opt.update(opt_update)
  return train_f(opt)

def extension(path):
  filename, file_extension = os.path.splitext(path)
  return file_extension

def loaddf(path):
  return pd.read_pickle(path)

def isopt(path):
  return extension(path) == ".pkl"


def isdf(path):
  return extension(path) == ".df"


"Loads all rundata files in `searchdir`"
def walkoptdata(searchdir):
  return walkload(searchdir, isopt, asl.load_opt)


"Loads all rundata files in `searchdir`"
def walkdfdata(searchdir):
  return walkload(searchdir, isdf, loaddf)


def walkload(searchdir, isgood, loaddata):
  data = []
  for root, dir, files in os.walk(searchdir, followlinks=True):
    for file in files:
      if isgood(file):
        print(root)
        data.append(loaddata(os.path.join(root, file)))

  return data

## Make figure 1
def nm_to_df(dfdata):
  nm_to_df_ = {}
  for df in dfdata:
    try:
      nm = df['runname'][0:1][0]
      nm_to_df_[nm] = df
    except:
      print("Missing data!")
  return nm_to_df_

def nm_to_opt(optdata):
  nm_to_opt_ = {}
  for opt in optdata:
    try:
      nm = opt['name']
      nm_to_opt_[nm] = opt
    except:
      print("Missing data!")
  return nm_to_opt_

# def optrunnames(optdata, dfdata):
#   optrunnames = []
#   for df in dfdata:
#     try:
#       optrunnames.append(df['runname'][0:1][0])
#     except:
#       print("Missing data!")
#   return optrunnames  

# Get all the data frames where the number of items is 1, find the optimal loss

def dfs_where_opt(filter_func, nm_to_df_, nm_to_opt_):
  dfs = []
  for nm, opt in nm_to_opt_.items():
    if filter_func(opt):
      dfs.append(nm_to_df_[nm])
  return dfs

def data(path):
  optdata = walkoptdata(path)
  dfdata = walkdfdata(path)
  nm_to_df_ = nm_to_df(dfdata)
  nm_to_opt_ = nm_to_opt(optdata)
  keys = set(list(nm_to_df_.keys())).intersection(set(list(nm_to_opt_.keys())))
  nm_to_df_ = {k:v for k,v in nm_to_df_.items() if k in keys}
  nm_to_opt_ = {k:v for k,v in nm_to_opt_.items() if k in keys}
  return nm_to_df_, nm_to_opt_

## Figure 1.a)
def optima_per_nitems(nm_to_df_, nm_to_opt_):
  optima = []
  for nitems in range(1, 11):
    dfs = dfs_where_opt(lambda opt: opt['nitems'] == nitems, nm_to_df_, nm_to_opt_)
    min_losses = [min(df['loss']) for df in dfs]
    print(min_losses)
    min_min_losses = min(min_losses)
    optima.append(min_min_losses)
  return optima

def fig1():
  nitems_vs_loss = optima_per_nitems(nm_to_df_, nm_to_opt_)

  plt.bar(np.arange(1, 11),
          nitems_vs_loss)
  plt.xlabel("Number of items")
  plt.ylabel("Mean square error")
  plt.title('Baseline Training Loss')

def runname(df):
  return df['runname'][0:1][0]

def capacity_trained_vs_tested(mat):
  plt.imshow(mat)
  plt.xlabel("Number of Items Trained on")
  plt.ylabel("Number of Items Tested on")
  plt.title("Stack Capacity Generalization")
  xinds = np.arange(1,11)
  yinds = reversed(xinds)
  plt.xticks(np.arange(10), xinds)
  plt.yticks(np.arange(10), yinds)
  cb = plt.colorbar()
  cb.set_label("Mean Square Error")

# def optimalloss(searchdir, nitems):
#   def nitemsisnitems(opt):
#     return opt["nitems"] == nitems
#   get_df_where_opt(nitemsisnitems)

# Figure 1.b) train capacity test
def optim_opt_df(nm_to_df_, nm_to_opt_, nitems):
  dfs = dfs_where_opt(lambda opt: opt['nitems'] == nitems, nm_to_df_, nm_to_opt_)
  min_losses = [min(df['loss']) for df in dfs]
  optim_id = min_losses.index(min(min_losses))
  optim_df = dfs[optim_id]
  optim_opt = nm_to_opt_[runname(optim_df)]
  return optim_opt, optim_df

def matrix_plot(nm_to_df_, nm_to_opt_):
  mat = np.zeros((10, 10))
  for i in range(1, 11):
    for j in range(1, 11):          
      optim_opt, optim_df = optim_opt_df(nm_to_df_, nm_to_opt_, i)
      res = record_data(optim_opt["log_dir"], {"nitems": j, "batch_size": 64})
      loss = res[2][0]["loss"].mean()
      mat[i - 1, j - 1] = loss
  return mat
  # res = record_data("/home/zenna/sshfs/omdata/runs/mnistsetsun/bsev_Feb04_18-37-19_node038_presuperbowl")



def capacity_trained_vs_tested_ax(mat, ax):
  im = ax.imshow(mat)
  ax.set_xlabel("Number of Items Trained on")
  ax.set_ylabel("Number of Items Tested on")
  ax.set_title("Stack Capacity Generalization")
  xinds = np.arange(1,11)
  yinds = reversed(xinds)
  ax.set_xticks(np.arange(10))
  ax.set_xticklabels(xinds)
  ax.set_yticks(np.arange(10))
  ax.set_yticklabels(yinds)
  return im 

def baseline_losses_ax(nm_to_df_, nm_to_opt_, ax):
  nitems_vs_loss = optima_per_nitems(nm_to_df_, nm_to_opt_)
  ax.bar(np.arange(1, 11) - 0.5,
          nitems_vs_loss,
          width=1,
          align="edge")
  ax.set_xticks(np.arange(10)+1)
  ax.set_xlim([0.5,10.5])
  ax.set_xlabel("Number of items")
  ax.set_ylabel("Mean square error")
  # ax.set_title('Baseline Training Loss')

## Figure 2.a) Loss curves
def loss_curve_for_item(nitems, nm_to_df_, nm_to_opt_, ax):
  for i in range(1, nitems + 1):
    _, optim_df =  optim_opt_df(nm_to_df_, nm_to_opt_, i)
    xs = optim_df["iteration"]
    ys = optim_df["loss"]
    ax.plot(xs, ys)
  ax.set_title("Training Loss vs Iteration")
  ax.set_ylabel("Mean Square Error")
  ax.set_xlabel("Iteration Number")
  ax.set_xscale('log')

if __name__ == "__main__":
  path = "/data/zenna/omruns/mnistsetsun"
  nm_to_df_, nm_to_opt_ = data(path)
  res = matrix_plot(nm_to_df_, nm_to_opt_)

  ## Figure 2
  fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(7, 9.6))
  im = capacity_trained_vs_tested_ax(np.rot90(mat), ax0)
  baseline_losses_ax(nm_to_df_, nm_to_opt_, ax1)
  plt.show()

  ## Figure 3
  fig1 = plt.figure()
  ax1 = fig1.add_subplot(111)
  loss_curve_for_item(10, nm_to_df_, nm_to_opt_, ax1)