"Utils common for benchmarking"
# from asl.reference import get_observes
import asl

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
  internals = log["{}/internal".format('reference')]
  for (j, internal) in enumerate(internals):
    writer.add_image('internals/{}'.format(j), internal.value[batch], i)
