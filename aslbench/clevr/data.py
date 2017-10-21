import os
from torch.utils.data import Dataset
from skimage import io
import asl.util as util

class ClevrData(Dataset):
  "Clevr DataSet"

  def __init__(self, root_dir, train=True, transform=None):
    self.train = "train" if train else "test"
    self.root_dir = root_dir
    datapath = os.path.join(root_dir, "images", self.train)
    self.nsamples = len([name for name in os.listdir(datapath) if os.path.isfile(name)])
    self.transform = transform
    self.prefix = os.path.join(datapath, "CLEVR_train_")

  def __len__(self):
    return self.nsamples

  def __getitem__(self, idx):
    img_name = os.path.join(self.prefix + '{0:06d}'.format(idx) + ".png")
    sample = io.imread(img_name)
    if self.transform:
        sample = self.transform(sample)
    return sample

root_dir = os.path.join(util.datadir(), "CLEVR_v1.0")
clvr = ClevrData(root_dir)
