import asl
from multipledispatch import dispatch
from torch import optim, nn

mnist_size = (1, 28, 28)

class Mnist(asl.Type):
  typesize = mnist_size

@dispatch(Mnist, Mnist)
def dist(x, y):
  return nn.MSELoss()(x.value, y.value)

def refresh_mnist(dl):
  "Extract image data and convert tensor to Mnist data type"
  return [asl.refresh_iter(dl, lambda x: Mnist(asl.util.image_data(x)))]
