import asl
from multipledispatch import dispatch
from torch import optim, nn
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch
from asl.util.io import datadir
import torchvision

mnist_size = (1, 28, 28)

class Mnist(asl.Type):
  typesize = mnist_size

@dispatch(Mnist, Mnist)
def dist(x, y):
  return nn.MSELoss()(x.value, y.value)

def refresh_mnist(dl, nocuda=False):
  "Extract image data and convert tensor to Mnist data type"
  return [asl.refresh_iter(dl, lambda x: Mnist(asl.util.image_data(x, nocuda=nocuda)))]

def mnistloader(batch_size, train=True, normalize=True):
  "Mnist data iterator"
  fs = [transforms.ToTensor()]
  if normalize:
    fs = fs + [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
  transform = transforms.Compose(fs)
  trainset = torchvision.datasets.MNIST(root=datadir(), train=train,
                                        download=True, transform=transform)
  return torch.utils.data.DataLoader(trainset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=1,
                                     drop_last=True)

MNIST.datatype = Mnist
MNIST.dataloader = mnistloader
MNIST.refresh = refresh_mnist