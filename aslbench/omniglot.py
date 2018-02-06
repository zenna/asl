import asl
from multipledispatch import dispatch
from torch import optim, nn

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
