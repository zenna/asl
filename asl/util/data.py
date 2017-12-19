import torchvision
import torchvision.transforms as transforms
import asl
from asl.util.io import datadir
import torch
from torch.autograd import Variable


def image_data(data):
  "Extract image data from mnist (and not classification)"
  return asl.cuda(Variable(data[0]))


def mnistloader(batch_size, train=True):
  "Mnist data iterator"
  transform = transforms.Compose(
  [transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  trainset = torchvision.datasets.MNIST(root=datadir(), train=train,
                                        download=True, transform=transform)
  return torch.utils.data.DataLoader(trainset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=1,
                                     drop_last=True)
