import os
from torch.utils.data import Dataset
from skimage import io
import asl.util as util


class ClevrImages(Dataset):
  "Clevr Imagaes DataSet"

  def __init__(self,
               clevr_root,
               train=True,
               transform=None):
    self.train = "train" if train else "test"
    self.clevr_root = clevr_root
    datapath = os.path.join(clevr_root, "images", self.train)
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


def clevr_iter(clevr_root,
               data_type,
               train=True):
  path = os.path.join(clevr_root, data_type)
  train_val = "train" if train else "val"
  if train:
    path = os.path.join(path, "CLEVR_{}_{}.json".format(train_val, data_type))
  else:
    path = os.path.join(path, "CLEVR_{}_{}.json".format(train_val, data_type))

  f = open(path, "r")
  return ijson.items(f, "{}.item".format(data_type))


def questions_iter(clevr_root=os.path.join(util.datadir(), "CLEVR_v1.0"),
                   train=True):
  "Iterator over question dataset"
  return clevr_iter(clevr_root, "questions", train)


def scenes_iter(clevr_root=os.path.join(util.datadir(), "CLEVR_v1.0"),
                train=True):
  "Iterator over scenes"
  return clevr_iter(clevr_root, "scenes", train)


def data_iter(batch_size, train=True):
  "Iterates paired scene and question data"
  qitr = questions_iter(train=train)
  sitr = scenes_iter(train=train)

  while True:
    rel_tens = []
    obj_set_tens = []
    progs = []
    answers = []
    for i in range(batch_size):
      qi = next(qitr)
      si = next(sitr)
      scenei = SceneGraph.from_json(si)
      rel_ten = scenei.relations.tensor()
      rel_tens.append(rel_ten)
      obj_ten = scenei.object_set.tensor()
      obj_set_tens.append(obj_ten)
      progs.append(qi['program'])
      answers.append(qi['answer'])

    yield progs, obj_set_tens, rel_tens, answers
