"Utils common for benchmarking"
# from asl.reference import get_observes
import asl
import os
from tensorboardX import SummaryWriter
from asl.callbacks import every_n
import common

def plot_observes(i, log, writer, batch=0, **kwargs):
  # import pdb; pdb.set_trace()
  if 'runstate' in log:
    observes = log['runstate']['observes']
    for mode in observes.keys():
      for label in observes[mode].keys():
        img = observes[mode][label].value[batch]
        writer.add_image('observes/{}/{}'.format(mode, label), img, i)

  # "Show the observed values in tensorboardX"
  # for k in observes.keys():
  #   refimg = log['ref_observes'][k].value
  #   neuimg = log['observes'][k].value
  #   writer.add_image('observes/{}/ref'.format(k), refimg[batch], i)
  #   writer.add_image('observes/{}/neural'.format(k), neuimg[batch], i)


def plot_empty(i, log, writer, **kwargs):
  "Show the empty set in tensorboardX"
  img = log['empty'][0].value
  writer.add_image('Empty', img, i)


def plot_internals(i, log, writer, batch=0, **kwargs):
  "Show internal structure. Shows anything log[NEURAL/internal]"
  internals = log["{}/internal".format('model')]
  for (j, internal) in enumerate(internals):
    writer.add_image('internals/{}'.format(j), internal.value[batch], i)

def trainloadsave(fname, train_fun, morerunoptsgen, custom_args):
  # Add stack-specific parameters to the cmdlargs
  cmdrunopt, dispatch_opt = asl.handle_args(custom_args)
  if dispatch_opt["dispatch"]:
    # import pdb; pdb.set_trace()
    morerunopts = morerunoptsgen(dispatch_opt["nsamples"])
    # Merge each runopt with command line opts (which take precedence)
    for opt in morerunopts:
      for k, v in cmdrunopt.items():
        if k not in opt:
          opt[k] = v
      # opt.update(cmdrunopt)

    asl.dispatch_runs(fname, dispatch_opt, morerunopts)
  else:
    if dispatch_opt["optfile"] is not None:
      keys = None if cmdrunopt["resume_path"] is None else ["arch", "lr"] 
      cmdrunopt.update(asl.load_opt(cmdrunopt["optfile"], keys))
      print("Loaded", cmdrunopt)
      print("resume_path", cmdrunopt["resume_path"] is None)
      return train_fun(cmdrunopt)
    else:
      asl.save_opt(cmdrunopt)
      return train_fun(cmdrunopt)

def trainmodel(opt, model, loss_gen, **trainkwargs):
  "The model"
  # Setup optimizer
  parameters = model.parameters()
  optimizer = opt["optimizer"](parameters, opt["lr"])
  asl.opt.save_opt(opt)
  if opt["resume_path"] is not None and opt["resume_path"] != '':
    asl.load_checkpoint(opt["resume_path"], model, optimizer)

  tbkeys = ["batch_size", "lr", "name", "nitems", "batch_norm", "nrounds"]
  optstring = asl.hyper.search.linearizeoptrecur(opt, tbkeys)
  if opt["train"]:
    writer = SummaryWriter(os.path.join(opt["log_dir"], optstring))
    update_df, save_df = asl.callbacks.save_update_df(opt)
    asl.train(loss_gen,
              optimizer,
              # maxiters=10,
              cont=asl.converged(1000),
              callbacks=[asl.print_loss(1),
                        every_n(common.plot_empty, 100),
                        every_n(common.plot_observes, 100),
                        every_n(common.plot_internals, 100),
                        every_n(asl.save_checkpoint(model), 1000),
                        every_n(save_df, 100),
                        update_df],
              post_callbacks=[save_df],
              log_dir=opt["log_dir"],
              writer = writer,
              **trainkwargs)
  return model, loss_gen