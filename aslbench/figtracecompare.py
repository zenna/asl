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
           (stacktraces.tracegen5, 2) # 10
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
    loss = optim_df["loss"].mean()
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

def trace_baseline_losses_ax(nm_to_df_, nm_to_opt_, ax):
  nitems_vs_loss = optima_losses(nm_to_df_, nm_to_opt_)
  ax.bar(np.arange(1, 11) - 0.5,
          nitems_vs_loss,
          width=1,
          align="edge")
  ax.set_xticks(np.arange(10)+1)
  ax.set_xlim([0.5,10.5])
  xinds = ["a1", "a2", "b1", "b2", "c1", "c2", "d1", "d2", "e1", "e2"]
  ax.set_xticklabels(xinds)
  ax.set_xlabel("Trace")
  ax.set_ylabel("Training Mean Square Error")
  # ax.set_title('Baseline Training Loss')

def traces_ax(mat, ax):
  im = ax.imshow(mat,
                 norm=colors.LogNorm(vmin=mat.min(), vmax=mat.max()))
  ax.set_xlabel("Trace Trained on")
  ax.set_ylabel("Trace Tested on")
  ax.set_title("Stack Traces Generalization")
  xinds = ["a1", "a2", "b1", "b2", "c1", "c2", "d1", "d2", "e1", "e2"]
  yinds = reversed(xinds)
  ax.set_xticks(np.arange(10))
  ax.set_xticklabels(xinds)
  ax.set_yticks(np.arange(10))
  ax.set_yticklabels(yinds)
  return im 

import matplotlib.colors as colors

if __name__ == "__main__":
  path = "/data/zenna/omruns/stacktracesx"
  optdata = walkoptdata(path)
  nm_to_df_ = nm_to_df_from_opt(optdata)
  nm_to_opt_ = nm_to_opt(optdata)
  nm_to_df_, nm_to_opt_ = intersect_df_opt(nm_to_df_, nm_to_opt_)
  mat = traces_matrix_plot(nm_to_df_, nm_to_opt_, 10)

fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(7, 9.6))
im = traces_ax(np.rot90(mat), ax0)
trace_baseline_losses_ax(nm_to_df_, nm_to_opt_, ax1)
# fig.colorbar(im)
plt.show()
