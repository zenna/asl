from stacktraces import *
from figures import *
import stacktraces
import itertools

traces =  [(stacktraces.tracegen1, 1), # 1
           (stacktraces.tracegen1, 2), # 2
           (stacktraces.tracegen2, 1), # 3
           (stacktraces.tracegen2, 2), # 4
           (stacktraces.tracegen3, 1), # 5 
           (stacktraces.tracegen3, 2), # 6
           (stacktraces.tracegen4, 1), # 7
           (stacktraces.tracegen4, 2), # 8
           (stacktraces.tracegen5, 1), # 9
           (stacktraces.tracegen5, 2), # 10
           (stacktraces.tracegen6, 1), # 9
           (stacktraces.tracegen6, 2) # 10
           ]

def optim_opt_df(f, nm_to_df_, nm_to_opt_):
  "Find the run, where f(opt) is true and df has min losses"
  dfs = dfs_where_opt(f, nm_to_df_, nm_to_opt_)
  min_losses = [min(df['loss']) for df in dfs]
  optim_id = min_losses.index(min(min_losses))
  optim_df = dfs[optim_id]
  optim_opt = nm_to_opt_[runname(optim_df)]
  return optim_opt, optim_df


def traces_matrix_plot(nm_to_df_, nm_to_opt_, n):
  mat = np.zeros((n, n))
  # import pdb; pdb.set_trace()
  for i, tracenr_train in enumerate(traces):
    for j, tracenr_test in enumerate(traces):
      trace_train, nr_train = tracenr_train
      trace_test, nr_test = tracenr_test
      def getopt(opt):
        return opt['tracegen'] == trace_train and opt['nrounds'] == nr_train
      optim_opt, optim_df = optim_opt_df(getopt, nm_to_df_, nm_to_opt_)
      res = record_data(train_stack, optim_opt["log_dir"], {"tracegen": trace_test,
                                                "nrounds": nr_test,
                                                "batch_size": 64,
                                                "nitems": 3,
                                                "test": True})
      loss = res[2][0]["loss"].mean()
      mat[i, j] = loss
  return mat

def optima_losses(nm_to_df_, nm_to_opt_):
  # import pdb; pdb.set_trace()
  optima = []
  for i, tracenr_train in enumerate(traces):
    trace_train, nr_train = tracenr_train
    def getopt(opt):
      return opt['tracegen'] == trace_train and opt['nrounds'] == nr_train
    optim_opt, optim_df = optim_opt_df(getopt, nm_to_df_, nm_to_opt_)
    loss = optim_df["loss"].min()
    optima.append(loss)
  return optima
  
def nm_to_df_from_opt(optdata):
  nm_to_df_ = {}
  for opt in optdata:
    try:
      res = record_data(train_stack, opt["log_dir"], {"batch_size": 64, "test": True})
      nm_to_df_[opt["name"]] = res[2][0]
    except:
      pass
  return nm_to_df_

def trace_baseline_losses_ax(nm_to_df_, nm_to_opt_, ax, n=12):
  nitems_vs_loss = optima_losses(nm_to_df_, nm_to_opt_)
  ax.bar(np.arange(1, n+1) - 0.5,
          nitems_vs_loss,
          width=1,
          align="edge")
  ax.set_xticks(np.arange(n)+1)
  ax.set_xlim([0.5,n + 0.5])
  xinds = ["a1", "a2", "b1", "b2", "c1", "c2", "d1", "d2", "e1", "e2", "f1", "f2"]
  ax.set_xticklabels(xinds)
  ax.set_xlabel("Trace")
  ax.set_ylabel("Training Mean Square Error")
  # ax.set_title('Baseline Training Loss')

def traces_ax(mat, ax, n = 12):
  im = ax.imshow(mat,
                 norm=colors.LogNorm(vmin=mat.min(), vmax=mat.max()))
  ax.set_xlabel("Trace Trained on")
  ax.set_ylabel("Trace Tested on")
  ax.set_title("Stack Traces Generalization")
  xinds = ["a1", "a2", "b1", "b2", "c1", "c2", "d1", "d2", "e1", "e2", "f1", "f2"]
  yinds = reversed(xinds)
  ax.set_xticks(np.arange(n))
  ax.set_xticklabels(xinds)
  ax.set_yticks(np.arange(n))
  ax.set_yticklabels(yinds)
  return im 

import matplotlib.colors as colors

def integrals_losses_ax(mat, ax, axis=0, n=12):
  integrals = np.sum(mat, axis=axis)
  ax.bar(np.arange(1, n+1) - 0.5,
          integrals,
          width=1,
          align="edge")
  ax.set_xticks(np.arange(n)+1)
  ax.set_xlim([0.5,n + 0.5])
  ax.set_title("Accumulative Test Loss")
  # ax.set_xlabel("Number of Items")
  ax.set_ylabel("MSE summed over tests")


def h_integrals_losses_ax(mat, ax, axis=1, n=12):
  integrals = np.sum(mat, axis=axis)
  integrals = list(reversed(integrals))
  ax.barh(np.arange(1, n+1) - 0.5,
          integrals,
          height=1,
          align="edge")
  ax.set_yticks(np.arange(n)+1)
  ax.set_ylim([0.5,n + 0.5])
  ax.set_title("Accumulative Test Loss")
  # ax.set_xlabel("Number of Items")
  ax.set_ylabel("MSE summed over tests")

def loss_curve_for_combo(nm_to_df_, nm_to_opt_, ax):
  # import pdb; pdb.set_trace()
  optima = []
  for i, tracenr_train in enumerate(traces):
    trace_train, nr_train = tracenr_train
    def getopt(opt):
      return opt['tracegen'] == trace_train and opt['nrounds'] == nr_train
    optim_opt, optim_df = optim_opt_df(getopt, nm_to_df_, nm_to_opt_)
    xs = optim_df["iteration"]
    ys = optim_df["loss"]
    ax.plot(xs, ys, linewidth=0.5)
  ax.set_title("Training Loss vs Iteration")
  ax.set_ylabel("MSE")
  # ax.set_xlabel("Iteration Number")
  ax.set_xscale('log')
  ax.set_xlim(1, 100000)

# if __name__ == "__main__":
  path = "/data/zenna/omruns/stacktracebeta"
  optdata = walkoptdata(path)
  dfdata = walkdfdata(path)
  nm_to_df_ = nm_to_df_from_opt(optdata)
  nm_to_opt_ = nm_to_opt(optdata)
  nm_to_df_ = nm_to_df(dfdata)
  nm_to_df_, nm_to_opt_ = intersect_df_opt(nm_to_df_, nm_to_opt_)
  mat = traces_matrix_plot(nm_to_df_, nm_to_opt_, 12)

def fig1():
  fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(7, 9.6))
  im = traces_ax(np.rot90(mat), ax0)
  trace_baseline_losses_ax(nm_to_df_, nm_to_opt_, ax1)
  fig.colorbar(im)
  plt.show()

def fig2(mat):
  plt.rcParams.update({'font.size': 4})
  mat = np.rot90(mat)
  fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(3, 3.6),
                         gridspec_kw={'height_ratios':[1,2,1]})
  ax00 = ax[0,0]
  ax01 = ax[0,1]
  ax10 = ax[1,0]
  ax11 = ax[1, 1]
  ax11 = ax[1, 1]
  ax20 = ax[2, 0]
  integrals_losses_ax(mat, ax00)
  h_integrals_losses_ax(mat, ax11, axis=1)
  im = traces_ax(mat, ax10)
  trace_baseline_losses_ax(nm_to_df_, nm_to_opt_, ax20)
  loss_curve_for_combo(nm_to_df_, nm_to_opt_, ax01)
  plt.subplots_adjust(hspace=0.6, wspace=0.3)
  plt.colorbar(im, ax=ax10)
  plt.show()


def datatest(traces, nm_to_df_, nm_to_opt_, datasets):
  results = []
  d1, d2 = datasets
  for trace in traces:
    for nr in [1, 2]:
      def getopt(opt):
        return opt['tracegen'] == trace and opt['dataset'] == d1 and opt['nrounds'] == nr
      optim_opt, optim_df = optim_opt_df(getopt, nm_to_df_, nm_to_opt_)
      res = record_data(train_stack, optim_opt["log_dir"], {"tracegen": trace,
                                              "dataset": d2,
                                              "batch_size": 64,
                                              "test": True})
      loss1 = optim_df["loss"].min()
      loss2 = res[2][0]["loss"].min()
      tp = [optim_opt, nr, loss1, loss2, d1, d2]
      results.append(tp)
  return results

trcs = [stacktraces.tracegen1,
        stacktraces.tracegen2,
        stacktraces.tracegen3,
        stacktraces.tracegen4,
        stacktraces.tracegen5,
        stacktraces.tracegen6]
datacompare = datatest(trcs, nm_to_df_, nm_to_opt_, ["omniglot", "mnist"])
datacompare2 = datatest(trcs, nm_to_df_, nm_to_opt_, ["mnist", "omniglot"])
datacompare3 = datatest(trcs, nm_to_df_, nm_to_opt_, ["mnist", "omniglot"])

def ok(vals, ax, shift):
  ind = np.arange(len(vals))  # the x locations for the groups
  width = 0.4       # the width of the bars
  if shift:
    indx = ind + width
  else:
    indx = ind
  return ax.bar(indx, vals, width, log=True)
  
def figure5(datacompare1, datacompare2, n=12):
  plt.rcParams.update({'font.size': 4})
  xinds = ["a1", "a2", "b1", "b2", "c1", "c2", "d1", "d2", "e1", "e2", "f1", "f2"]
  fig, ax = plt.subplots(ncols=2, figsize=(3.2, 1.6))
  axl, axr = ax
  train_losses = [dt[2]  for dt in datacompare1]
  test_losses = [dt[3]  for dt in datacompare1]
  rects1 = ok(test_losses, axl, True)
  rects0 = ok(train_losses, axl, False)

  train_losses = [dt[2]  for dt in datacompare2]
  test_losses = [dt[3]  for dt in datacompare2]
  rects1b = ok(test_losses, axr, True)
  rects0b = ok(train_losses, axr, False)
  axl.legend((rects0, rects1), ('Train', 'Test'))
  axl.set_xticks(np.arange(n)+0.2)
  # axl.set_xlim([0.5,n + 0.5])
  axl.set_xticklabels(xinds)

  axr.legend((rects0, rects1), ('Train', 'Test'))
  axr.set_xticks(np.arange(n)+0.2)
  # axr.set_xlim([0.5,n + 0.5])
  axr.set_xticklabels(xinds)
  axl.set_title("Train: Mnist, Test :Omniglot")


  axl.set_xlabel("Trace")
  axr.set_xticklabels(xinds)
  axr.set_xlabel("Trace")
  axr.set_title("Train: Omniglot, Test: Mnist")


  


figure5(datacompare, datacompare2)