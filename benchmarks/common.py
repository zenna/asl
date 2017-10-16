"Utils common for benchmarking"
from asl.sketch import Mode

def plot_observes(i, log, writer, batch=0, **kwargs):
  "Show the empty set in tensorboardX"
  for j in range(len(log['observes'])):
    writer.add_image('comp{}/ref'.format(j), log['ref_observes'][j][batch], i)
    writer.add_image('comp{}/neural'.format(j), log['observes'][j][batch], i)


def plot_empty(i, log, writer, **kwargs):
  "Show the empty set in tensorboardX"
  img = log['empty'][0].value
  writer.add_image('Empty', img, i)


def plot_internals(i, log, writer, batch=0, **kwargs):
  "Show internal structure"
  internals = log["{}/internal".format(Mode.NEURAL.name)]
  for (j, internal) in enumerate(internals):
    writer.add_image('internals/{}'.format(j), internal[batch], i)
