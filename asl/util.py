import matplotlib.pyplot as plt
plt.ion()
import torchvision
import torchvision.transforms as transforms
import torch

def draw(t):
  "Draw a tensor"
  tnp = t.data.cpu().numpy().squeeze()
  plt.imshow(tnp)
  plt.pause(0.01)


def trainloader(batch_size):
  transform = transforms.Compose(
  [transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
  return torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        shuffle=False, num_workers=1)
