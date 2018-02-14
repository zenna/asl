"Learning a language"
import random
import argparse
import asl
from asl.modules.modules import ConstantNet, ModuleDict, expand_to_batch
from aslbench.clevr.data import clevr_img_dl
from torch import optim, nn
import torch
from torch.autograd import Variable
import common
from multipledispatch import dispatch
import numpy as np
from primitive import Integer, IntegerOneHot1D


## datasets
from clevrimage import clevr_img_size, ClevrImage
from mnist import mnist_size, Mnist, dist, refresh_mnist
from omniglot import omniglot_size, OmniGlot, dist, refresh_omniglot

# SMALL_IMG_SIZE = clevr_img_size
# Image = ClevrImage
nimages = 2     # NUmber of images in set

def lang_args(parser):
  parser.add_argument('--nimages', type=int, default=3, metavar='NI',
                      help='number of images in set (default: 3)')


def train_clevrlang(opt):
  if opt["dataset"] == "omniglot":
    sentence_size = asl.util.repl(mnist_size, 0, opt["nchannels"])
    ImageType = OmniGlot
    dataloader = asl.util.omniglotloader
    refresh_data = lambda dl: refresh_omniglot(dl, nocuda=opt["nocuda"])
  elif opt["dataset"] == "mnist":
    sentence_size = asl.util.repl(mnist_size, 0, opt["nchannels"])
    ImageType = Mnist
    dataloader = asl.util.mnistloader
    refresh_data = lambda dl: refresh_mnist(dl, nocuda=opt["nocuda"])
  else:
    assert False

  ## TYPES
  ## =====
  class Sentence(asl.Type):
    "A sentence in a language (describing an image)"
    typesize = sentence_size # HACK

  ## Functions
  ## =========
  class Describe(asl.Function, asl.Net):
    "Describe the ith image of an image set"
    def __init__(self, ImageType, name="Describe", **kwargs):
      asl.Function.__init__(self, [ImageType for _ in range(nimages)] + [Integer],
                                  [Sentence])
      asl.Net.__init__(self, name, **kwargs)

  class WhichImage(asl.Function, asl.Net):
    "Determine which image the sentence describes"
    def __init__(self, ImageType, name="WhichImage", **kwargs):
      asl.Function.__init__(self, [ImageType for _ in range(nimages)] + [Sentence],
                                  [Integer])
      asl.Net.__init__(self, name, **kwargs)

  describe = Describe(ImageType, arch=opt["arch"], arch_opt=opt["arch_opt"])
  which_image = WhichImage(ImageType, arch=opt["arch"], arch_opt=opt["arch_opt"])

  class PermuteTest(asl.Sketch):
    """Generate clevr image from noise"""

    def sketch(self, imagesgen, r, runstate):
      # imgj = images[rand_img_id]
      images = [next(imagesgen) for _ in range(nimages)]
      permutation = list(range(nimages))
      r.shuffle(permutation)
      rand_img_id = r.randint(0, nimages)
      rand_img_id_1hot = IntegerOneHot1D(Variable(asl.util.cuda(asl.util.onehot(rand_img_id, 3, 1))))
      (sentence, ) = describe(*images, rand_img_id_1hot)
      permuted_images = [images[i] for i in permutation]
      (score, ) = which_image(*permuted_images, sentence)
      asl.observe(score, "score", runstate)
      return (score, )
    
  def target_score_gen(images, r, runstate):
    permutation = list(range(nimages))
    r.shuffle(permutation)
    rand_img_id = r.randint(0, nimages)
    score = permutation[rand_img_id]
    score_1hot = IntegerOneHot1D(Variable(asl.util.cuda(asl.util.onehot(score, 3, 1))))
    asl.observe(score_1hot, "score", runstate)
    return (score_1hot, )
  
  permute_test = PermuteTest([ImageType for _ in range(nimages)], [Integer])

  # Cuda everything
  interface = ModuleDict({"describe": describe,
                          "which_image": which_image})
  asl.cuda(interface, opt["nocuda"])

  # Generators CLEVR img generator
  img_dl = dataloader(opt["batch_size"], normalize=False)
  # chooseimage = infinitesamples[asl.onehot(3, 4, 20)]

  # Hack to add random object as input to traces
  def refresh_with_random(x):
    refreshed = refresh_data(x)
    return [refreshed[0], random.Random(0)]


  # Loss
  loss_gen = asl.ref_loss_gen(permute_test,
                              target_score_gen,
                              img_dl,
                              refresh_with_random)
  parameters = interface.parameters()
  return common.trainmodel(opt, interface, loss_gen, parameters)