import torchvision
import torchvision.transforms as transforms
import asl
from asl.util.io import datadir
import torch
from torch.autograd import Variable

def train_data(data):
  return asl.util.misc.cuda(Variable(data[0]))


def trainloader(batch_size):
  transform = transforms.Compose(
  [transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  trainset = torchvision.datasets.MNIST(root=datadir(), train=True,
                                        download=True, transform=transform)
  return torch.utils.data.DataLoader(trainset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=1,
                                     drop_last=True)
