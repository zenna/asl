import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 12
output_object_size = (batch_size, 10, 4, 8)
relations_size = (batch_size, 4, 10, 10)

obj_size = 10 * 4 * 8
relations_size = 4 * 10 * 10
total_output = obj_size + relations_size

imagedata = torch.autograd.Variable(torch.rand(batch_size, 3, 480, 320))
x = nn.Conv2d(3, 8, 3, padding=3)(imagedata)
x = nn.Conv2d(3, 8, 3, padding=3)(imagedata)
x = nn.MaxPool2d(5)(x)
x = nn.Conv2d(8, 1, 3, padding=3)(x)
x = nn.MaxPool2d(3)(x)
x = x.view(-1, 33*22)
x = nn.Linear(33*22, total_output)(x)


x = nn.Conv2d(8, 8, 3, padding=3)
obj_data = x[:, 0:obj_size]
object_prediction = obj_data.contiguous().view(-1, 10, 4, 8)
rel_data = x[:, obj_size:obj_size + relations_size]

import torch.utils.data.Dataset
class ClevrData(Dataset):
  "Clevr DataSet"

  def __init__(self, root_dir, transform=None):
    self.root_dir = root_dir
    datapath = os.path.join(root_dir, "train")
    self.nsamples = len([name for name in os.listdir(datapath) if os.path.isfile(name)])

  def __len__(self):
    return self.nsamples

class Vision(nn.Module):

  def __init__(self, nhlayers=2):
    nn.Module.__init__(self)
    self.l1 = nn.Conv2d(3, 8, 3, padding=3)
    self.l2 = nn.Conv2d(8, 8, 3, padding=3)
    self.l3 = nn.Conv2d(8, 1, 3, padding=3)
    self.l4 = nn.Linear(33 * 23, total_output)

  def forward(self, input_img):
    import pdb; pdb.set_trace()
    x = self.l1(input_img)
    x = F.elu(x)
    x = self.l2(x)
    x = F.elu(x)
    x = nn.MaxPool2d(5)(x)
    x = self.l3(x)
    x = F.elu(x)
    x = nn.MaxPool2d(3)(x)
    x = x.view(-1, 33 * 23)
    x = self.l4(x)
    x = F.sigmoid(x)

    obj_data = x[:, 0:obj_size]
    object_prediction = obj_data.contiguous().view(-1, 10, 4, 8)
    rel_data = x[:, obj_size:]
    rel_data = rel_data.contiguous().view(-1, 4, 10, 10)
    return obj_data, rel_data

myconvnet = Vision()
res = myconvnet(imagedata)
optimizer = torch.optim.Adam(myconvnet.parameters())
lossf = torch.nn.MSELoss()

for i in range(10000):
  random_input_img = torch.autograd.Variable(torch.rand(batch_size, 3, 480, 320))
  output = myconvnet(random_input_img)
  fake_target = torch.autograd.Variable(torch.rand(12, 8, 492,332))
  loss = lossf(output, fake_target)
  print("Loss is ", loss)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
