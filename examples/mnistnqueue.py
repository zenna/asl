from asl.type import Type
from asl.callbacks import tb_loss, every_n, print_loss
from asl.util.misc import trainloader, iterget, train_data
from asl.util.io import handle_args
from asl.log import log_append
from asl.train import train
from asl.structs.nstack import neural_stack, ref_stack

from torch import optim, nn


def stack_trace(items, push, pop, empty):
  """Example stack trace"""
  log_append("empty", empty)

  observes = []
  stack = empty
  # print("mean", items[0].mean())
  (stack,) = push(stack, items[0])
  (stack,) = push(stack, items[1])
  # (stack,) = push(stack, items[2])
  (pop_stack, pop_item) = pop(stack)
  observes.append(pop_item)
  (pop_stack, pop_item) = pop(pop_stack)
  observes.append(pop_item)
  log_append("observes", observes)
  log_append("empty", empty)
  return observes


def plot_observes(i, log, writer, **kwargs):
  "Show the empty set in tensorboardX"
  refobserves = log['observes'][0]
  nobserves = log['observes'][1]
  for j in range(len(refobserves)):
    writer.add_image('compare{}/ref'.format(j), refobserves[j][0], i)
    writer.add_image('compare{}/neural'.format(j), nobserves[j][0], i)


def plot_empty(i, log, writer, **kwargs):
  "Show the empty set in tensorboardX"
  img = log['empty'][0].value
  writer.add_image('EmptySet', img, i)


def observe_loss(criterion, obs, refobs, state=None):
  "MSE between observations from reference and training stack"
  total_loss = 0.0
  losses = [criterion(obs[i], refobs[i]) for i in range(len(obs))]
  total_loss = sum(losses)
  return total_loss


def train_stack():
  options = handle_args()
  nitems = 3
  mnist_size = (1, 28, 28)

  class MatrixStack(Type):
    size = mnist_size

  class Mnist(Type):
    size = mnist_size

  tl = trainloader(options.batch_size)
  items_iter = iter(tl)
  ref_items_iter = iter(tl)
  nstack = neural_stack(MatrixStack, Mnist)
  refstack = ref_stack()

  criterion = nn.MSELoss()
  optimizer = optim.Adam(nstack.parameters(), lr=options.lr)

  def loss_gen():
    nonlocal items_iter, ref_items_iter

    try:
      items = iterget(items_iter, nitems, transform=train_data)
      ref_items = iterget(ref_items_iter, nitems, transform=train_data)
    except StopIteration:
      print("End of Epoch")
      items_iter = iter(tl)
      ref_items_iter = iter(tl)
      items = iterget(items_iter, nitems, transform=train_data)
      ref_items = iterget(ref_items_iter, nitems, transform=train_data)

    observes = stack_trace(items, **nstack)
    refobserves = stack_trace(ref_items, **refstack)
    return observe_loss(criterion, observes, refobserves)

  train(loss_gen, optimizer, maxiters=100000,
        callbacks=[print_loss(100), plot_empty, plot_observes])

if __name__ == "__main__":
  train_stack()
