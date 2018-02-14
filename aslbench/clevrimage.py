import asl
from multipledispatch import dispatch
from torch import optim, nn

clevr_img_size = (3, 80, 120)

class ClevrImage(asl.Type):
  typesize = clevr_img_size

def refresh_clevr(dl, nocuda=False):
  "Extract image data and convert tensor to Clevr data type"
  return [asl.refresh_iter(dl, lambda x: ClevrImage(asl.util.image_data(x, nocuda=nocuda)))]
