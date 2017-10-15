import sys
from asl.type import Type, Function
from asl.callbacks import tb_loss, every_n
from asl.util.misc import draw, trainloader, as_img, iterget
from asl.util.data import train_data
from asl.util.io import handle_args
from asl.log import log_append
from asl.train import train
from asl.structs.nstack import neural_stack, ref_stack

from torch.autograd import Variable
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
    return observes


def plot_observes(i, log, writer, **kwargs):
  "Show the empty set in tensorboardX"
  ref_observes = log['observes'][0]
  nobserves = log['observes'][1]
  for j in range(len(ref_observes)):
    writer.add_image('compare{}/ref'.format(j), ref_observes[j][0], i)
    writer.add_image('compare{}/neural'.format(j), nobserves[j][0], i)


def plot_empty(i, log, writer, **kwargs):
  "Show the empty set in tensorboardX"
  img = log['empty'][0][0]
  writer.add_image('EmptySet', img, i)


def observe_loss(criterion, obs, refobs, state=None):
  "MSE between observations from reference and training stack"
  total_loss = 0.0
  losses = [criterion(obs[i], refobs[i]) for i in range(len(obs))]
  print([loss[0].data[0] for loss in losses])
  total_loss = sum(losses)
  return total_loss


def test_stack():
  options = handle_args()
  print("Using CUDA", options.cuda)
  tl = trainloader(options.batch_size)
  items_iter = iter(tl)
  ref_items_iter = iter(tl)
  matrix_stack = Type("Stack", (1, 28, 28), dtype="float32")
  mnist_type = Type("mnist_type", (1, 28, 28), dtype="float32")
  nstack = neural_stack(mnist_type, matrix_stack)
  refstack = ref_stack()

  criterion = nn.MSELoss()
  optimizer = optim.Adam(nstack.parameters(), lr=options.lr)

  def loss_gen():
    nonlocal items_iter, ref_items_iter

    try:
      items = iterget(items_iter, 3, transform=train_data)
      ref_items = iterget(ref_items_iter, 3, transform=train_data)
    except StopIteration:
      print("End of Epoch")
      items_iter = iter(tl)
      ref_items_iter = iter(tl)
      items = iterget(items_iter, 3, transform=train_data)
      ref_items = iterget(ref_items_iter, 3, transform=train_data)

    observes = stack_trace(items, **nstack)
    ref_observes = stack_trace(ref_items, **refstack)
    return observe_loss(criterion, observes, ref_observes)

  train(loss_gen, optimizer)

if __name__ == "__main__":
  test_stack()
