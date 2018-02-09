from mnistset import record_data

import itertools
import asl
import pandas as pd
import os
import stdargs
import numpy as np
import torch
# import matplotlib.pyplot as plt


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
  plt.xlabel("Number of Items")
  plt.ylabel("MSE")
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
  cb.set_label("MSE")

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

if __name__ == "__main__":
  path = "/data/zenna/omruns/mnistsetsun"
  nm_to_df_, nm_to_opt_ = data(path)
  res = matrix_plot(nm_to_df_, nm_to_opt_)

def capacity_trained_vs_tested_ax(mat, ax):
  im = ax.imshow(mat)
  # ax.set_xlabel("Number of Items Trained on")
  ax.set_ylabel("Number of Items Tested on")
  ax.set_title("Stack Capacity Generalization")
  xinds = np.arange(1,11)
  yinds = reversed(xinds)
  ax.set_xticks(np.arange(10))
  ax.set_xticklabels(xinds)
  ax.set_yticks(np.arange(10))
  ax.set_yticklabels(yinds)
  return im 

def integrals_losses_ax(mat, ax, axis=0):
  integrals = np.sum(mat, axis=axis)
  ax.bar(np.arange(1, 11) - 0.5,
          integrals,
          width=1,
          align="edge")
  ax.set_xticks(np.arange(10)+1)
  ax.set_xlim([0.5,10.5])
  ax.set_title("Accumulative Test Loss")
  # ax.set_xlabel("Number of Items")
  ax.set_ylabel("MSE summed over tests")


def h_integrals_losses_ax(mat, ax, axis=1):
  integrals = np.sum(mat, axis=axis)
  integrals = list(reversed(integrals))
  ax.barh(np.arange(1, 11) - 0.5,
          integrals,
          height=1,
          align="edge")
  ax.set_yticks(np.arange(10)+1)
  ax.set_ylim([0.5,10.5])
  ax.set_title("Accumulative Test Loss")
  # ax.set_xlabel("Number of Items")
  ax.set_ylabel("MSE summed over tests")

def baseline_losses_ax(nm_to_df_, nm_to_opt_, ax):
  nitems_vs_loss = optima_per_nitems(nm_to_df_, nm_to_opt_)
  ax.bar(np.arange(1, 11) - 0.5,
          nitems_vs_loss,
          width=1,
          align="edge")
  ax.set_xticks(np.arange(10)+1)
  ax.set_xlim([0.5,10.5])
  ax.set_title("Baseline Training Loss")
  ax.set_ylabel("Train MSE")
  ax.set_xlabel("Number of Items Trained on")

  # ax.set_title('Baseline Training Loss')


# fig.subplots_adjust(right=0.8)
# fig.colorbar(im)

  # cb = plt.colorbar()
  # cb.set_label("MSE")


## Figure 2.a) Loss curves
def loss_curve_for_item(nitems, nm_to_df_, nm_to_opt_, ax):
  for i in range(1, nitems + 1):
    _, optim_df =  optim_opt_df(nm_to_df_, nm_to_opt_, i)
    xs = optim_df["iteration"]
    ys = optim_df["loss"]
    ax.plot(xs, ys, linewidth=0.5)
  ax.set_title("Training Loss vs Iteration")
  ax.set_ylabel("MSE")
  # ax.set_xlabel("Iteration Number")
  ax.set_xscale('log')
  ax.set_xlim(1, 100000)

from matplotlib import gridspec

def fig1():
  mat = np.rot90(res)
  fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(2.6, 3),
                         gridspec_kw={'height_ratios':[1,2,1]})
  ax00 = ax[0,0]
  ax01 = ax[0,1]
  ax10 = ax[1,0]
  ax11 = ax[1, 1]
  ax11 = ax[1, 1]
  ax20 = ax[2, 0]
  integrals_losses_ax(mat, ax00)
  h_integrals_losses_ax(mat, ax11, axis=1)
  im = capacity_trained_vs_tested_ax(mat, ax10)
  baseline_losses_ax(nm_to_df_, nm_to_opt_, ax20)
  loss_curve_for_item(10, nm_to_df_, nm_to_opt_, ax01)
  plt.subplots_adjust(hspace=0.6, wspace=0.3)
  plt.colorbar(im, ax=ax10)
  plt.rcParams.update({'font.size': 4})
  plt.show()


def optima_per_consts(nm_to_df_, nm_to_opt_, init, lc):
  def getopt(opt):
    return opt["learn_constants"] == lc and opt["init"] == init

  dfs = dfs_where_opt(getopt, nm_to_df_, nm_to_opt_)
  min_losses = [min(df['loss']) for df in dfs]
  # import pdb; pdb.set_trace()
  min_min_losses = np.min(min_losses)
  std_losses = np.std(min_losses)
  median_losses = np.median(min_losses)
  mean_losses = np.mean(min_losses)
  return {"mean": mean_losses,
          "median": median_losses,
          "std": std_losses,
          "min": min_min_losses}

inits = [torch.zeros_like, torch.ones_like, torch.nn.init.normal, torch.nn.init.uniform]
init_names = ["zeros", "ones", "normal", "uniform"]
learn_constants = [True, False]

results = []
for (init, lc) in itertools.product(inits, learn_constants):
  resa = optima_per_consts(nm_to_df_, nm_to_opt_, init, lc)
  results.append((init, lc, resa))

def ok(vals, ax, shift):
  ind = np.arange(len(vals))  # the x locations for the groups
  width = 0.4       # the width of the bars
  if shift:
    indx = ind + width
  else:
    indx = ind
  return ax.bar(indx, vals, width, log=True)
  

def fig3(t = "min"):
  learn_mins = [resa[2][t] for i, resa in enumerate(results) if i % 2 ==0]
  nolearn_mins = [resa[2][t] for i, resa in enumerate(results) if i % 2 ==1]

  fig, ax = plt.subplots()
  # ax.set_yscale('log')
  rects0 = ok(learn_mins, ax, False)
  rects1 = ok(nolearn_mins, ax, True)
  # ax.set_xlabel()
  ax.set_title("Mean Loss")
  ax.set_xticks([0.2, 1.2, 2.2, 3.2])
  ax.set_xticklabels(["zeros", "ones", "normal", "uniform"])
  ax.set_ylabel("MSE")
  ax.legend((rects0, rects1), ('Learned', 'Constant'))
  plt.show()

fig3("min")