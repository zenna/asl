"Utils common for benchmarking"

def plot_observes(i, log, writer, **kwargs):
  "Show the empty set in tensorboardX"
  batch = 0
  for j in range(len(log['observes'])):
    writer.add_image('comp{}/ref'.format(j), log['observes'][j][batch], i)
    writer.add_image('comp{}/neural'.format(j), log['ref_observes'][j][batch], i)


def plot_empty(i, log, writer, **kwargs):
  "Show the empty set in tensorboardX"
  img = log['empty'][0].value
  writer.add_image('EmptySet', img, i)
