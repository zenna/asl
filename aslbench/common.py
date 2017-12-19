"Utils common for benchmarking"
# from asl.reference import get_observes
import asl

# def plot_observes(i, log, writer, batch=0, **kwargs):
#   "Show the observed values in tensorboardX"
#   for k in log['observes'].keys():
#     refimg = log['ref_observes'][k].value
#     neuimg = log['observes'][k].value
#     writer.add_image('observes/{}/ref'.format(k), refimg[batch], i)
#     writer.add_image('observes/{}/neural'.format(k), neuimg[batch], i)


def plot_empty(i, log, writer, **kwargs):
  "Show the empty set in tensorboardX"
  img = log['empty'][0].value
  writer.add_image('Empty', img, i)


# def log_observes(i, log, writer, **kwargs):
#   # import pdb; pdb.set_trace()
#   "Log observed values"
#   asl.log("observes", get_observes())
#   asl.log("ref_observes", get_ref_observes())


def plot_internals(i, log, writer, batch=0, **kwargs):
  "Show internal structure. Shows anything log[NEURAL/internal]"
  internals = log["{}/internal".format('reference')]
  for (j, internal) in enumerate(internals):
    writer.add_image('internals/{}'.format(j), internal.value[batch], i)
