import os
import torchvision
import torchvision.transforms as transforms
import asl
from asl.util.io import datadir
import torch
from torch.autograd import Variable


def image_data(data, nocuda=False):
  "Extract image data from mnist (and not classification)"
  return asl.cuda(Variable(data[0]), nocuda)


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

def omniglotloader(batch_size, background=True):
  "Omni-Glot data iterator"
  fs = [torchvision.transforms.Resize((28, 28)), transforms.ToTensor()]
  transform = transforms.Compose(fs)
  path = os.path.join(datadir(), "omniglot")
  dataset = asl.datasets.omniglot.Omniglot(path,
                                           download=True,
                                           transform=transform,
                                           background=background)
  return torch.utils.data.DataLoader(dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=1,
                                    drop_last=True)
