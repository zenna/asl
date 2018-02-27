"Utils common for benchmarking"
# from asl.reference import get_observes
import asl
import os
from tensorboardX import SummaryWriter
from asl.callbacks import every_n
import common
import numpy as np
import imageio
import pandas as pd

def save_img(path):
  x = 3

def write_img(img_batch, batch, prefix, logdir, i):
  img = img_batch[batch]
  nchannels = img.size(1)
  if nchannels > 3:
    img = img[0:3, :, :]
  fn = "{}_{}".format(prefix, i)  
  fn = os.path.join(logdir, fn)
  npimg = asl.util.as_img(img)
  np.save(fn, npimg)
  imageio.imwrite('{}.png'.format(fn), npimg)

def plot_observes(i, log, writer, batch=0, **kwargs):
  if 'runstate' in log:
    observes = log['runstate']['observes']
    for mode in observes.keys():
      for label in observes[mode].keys():
        img = observes[mode][label].value[batch]
        nchannels = img.size(0)
        if nchannels > 3:
          img = img[0:3, :, :]
        writer.add_image('observes/{}/{}'.format(mode, label), img, i)
        write_img(observes[mode][label].value,
                  batch,
                  'observes_{}_{}'.format(mode, label),
                  kwargs["log_dir"],
                  i)


def plot_empty(i, log, writer, **kwargs):
  "Show the empty set in tensorboardX"
  img = log['empty'][0].value
  nchannels = img.size(1)
  if nchannels > 3:
    img = img[:,0:3, :, :]
  writer.add_image('Empty', img, i)
  write_img(img,
            0,
            "Empty",
            kwargs["log_dir"],
            i)

def plot_internals(i, log, writer, batch=0, **kwargs):
  "Show internal structure. Shows anything log[NEURAL/internal]"
  internals = log["{}/internal".format('model')]
  for (j, internal) in enumerate(internals):
    img = internal.value[batch]
    nchannels = img.size(0)
    if nchannels > 3:
      img = img[0:3, :, :]
    writer.add_image('internals/{}'.format(j), img, i)
    write_img(internal.value,
              batch,
              "internals_{}".format(j),
              kwargs["log_dir"],
              i)

def trainloadsave(fname, train_fun, morerunoptsgen, custom_args):
  # Add custom parameters to the cmdlargs
  cmdrunopt, dispatch_opt = asl.handle_args(custom_args)
  morerunopts = morerunoptsgen(dispatch_opt["nsamples"])
  # Merge each runopt with command line opts (which take precedence)
  for opt in morerunopts:
    for k, v in cmdrunopt.items():
      if k not in opt:
        opt[k] = v
  if dispatch_opt["dispatch"]:
    asl.dispatch_runs(fname, dispatch_opt, morerunopts)
  else:
    if dispatch_opt["optfile"] is not None:
      keys = None if cmdrunopt["resume_path"] is None else ["arch", "lr"] 
      cmdrunopt.update(asl.load_opt(cmdrunopt["optfile"], keys))
      print("Loaded", cmdrunopt)
      print("resume_path", cmdrunopt["resume_path"] is None)
      return train_fun(cmdrunopt)
    else:
      # import pdb; pdb.set_trace()
      cmdrunopt = morerunopts[0]
      asl.save_opt(cmdrunopt)
      return train_fun(cmdrunopt)

def trainmodel(opt, model, loss_gen, parameters = None, **trainkwargs):
  "The model"
  # Setup optimizer
  parameters = model.parameters() if parameters is None else parameters
  optimizer = opt["optimizer"](parameters, opt["lr"])
  asl.opt.save_opt(opt)
  if opt["resume_path"] is not None and opt["resume_path"] != '':
    asl.load_checkpoint(opt["resume_path"], model, optimizer)

  tbkeys = ["activation",
            "nrounds",
            "nitems",
            "batch_size",
            "batch_norm",
            "dataset",
            "init",
            "ks",
            "learn_batch_norm",
            "lr",
            "name",
            "nchannels",
            "tracegen"]
  optstring = asl.hyper.search.linearizeoptrecur(opt, tbkeys)
  if opt["train"]:
    writer = SummaryWriter(os.path.join(opt["log_dir"], optstring))
    update_df, save_df = asl.callbacks.save_update_df(opt)

    callbacks = [asl.print_loss(100),
                 every_n(common.plot_empty, 1000),
                 every_n(common.plot_observes, 1000),
                 every_n(common.plot_internals, 1000),
                 every_n(asl.save_checkpoint(model), 1000),
                 every_n(save_df, 500),
                 update_df]

    if "test" in opt and opt["test"]:
      test_df = pd.DataFrame({'runname': [],
                              'iteration': [],
                              'loss': []})
      test_dfs = [test_df]
      dfcb = asl.callbacks.update_ret_df(opt, test_dfs)
      callbacks.append(dfcb)
    asl.train(loss_gen,
              optimizer,
              maxiters=1000,
              # cont=asl.convergedmin(1000),
              callbacks=callbacks,
              post_callbacks=[save_df],
              log_dir=opt["log_dir"],
              writer = writer,
              optimize=True,
              **trainkwargs)
  return model, loss_gen