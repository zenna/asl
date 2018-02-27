import os
import asl
from multipledispatch import dispatch
from torch import optim, nn
from asl.datasets.omniglot import Omniglot
import torchvision
import torchvision.transforms as transforms
from asl.util.io import datadir
import torch

# omniglot_size = (1, 105, 105)
omniglot_size = (1, 28, 28)

class OmniGlot(asl.Type):
  typesize = omniglot_size

@dispatch(OmniGlot, OmniGlot)
def dist(x, y):
  return nn.MSELoss()(x.value, y.value)

def refresh_omniglot(dl, nocuda=False): #SWITCHME FIXME
    "Extract image data and convert tensor to Mnist data type"
    return [asl.refresh_iter(dl, lambda x: OmniGlot(asl.util.image_data(x, nocuda=nocuda)))]

def omniglotloader(batch_size, background=True, normalize=True):
    "Omni-Glot data iterator"
    fs = [torchvision.transforms.Resize((28, 28)),
          transforms.ToTensor()]
    if normalize:
        fs = fs + [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
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

Omniglot.datatype = OmniGlot
Omniglot.dataloader = omniglotloader
Omniglot.refresh = refresh_omniglot