"Miscellaneous Utilities"
import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from itertools import islice
from torch.autograd import Variable

plt.ion()


def cuda(tensor, docuda=True):
  if docuda:
    return tensor.cuda()
  else:
    return tensor

def as_img(t):
  return t.data.cpu().numpy().squeeze()


def datadir(default='./data', varname='DATADIR'):
  "Data directory"
  if varname in os.environ:
    return os.environ['DATADIR']
  else:
    return default


def draw(t):
  "Draw a tensor"
  tnp = as_img(t)
  plt.imshow(tnp)
  plt.pause(0.01)


def trainloader(batch_size):
  transform = transforms.Compose(
  [transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  trainset = torchvision.datasets.MNIST(root=datadir(), train=True,
                                        download=True, transform=transform)
  return torch.utils.data.DataLoader(trainset,
                                     batch_size=batch_size,
                                     shuffle=False, num_workers=1,
                                     drop_last=True)

def iterget(dataiter, n):
  return [Variable(cuda(next(dataiter)[0])) for i in range(n)]

def is_tensor_var(tensor):
  "Is tensor either a tensor or variable?"
  return isinstance(tensor, torch.autograd.Variable) or torch.is_tensor(tensor)