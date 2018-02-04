"Learn a generative model of and from Clevr images"
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

def clevrlang_args(parser):
  parser.add_argument('--batch_norm', action='store_true', default=True,
                      help='Do batch norm')
  parser.add_argument('--num_images', action='store_true', default=3,
                      help='Number o fimages in set')

def clevrlang_args_sample():
  "Options sampler"
  return argparse.Namespace(batch_norm=np.random.rand() > 0.5)

CLEVR_IMG_SIZE = (3, 80, 120)
SMALL_IMG_SIZE = CLEVR_IMG_SIZE
# SMALL_IMG_SIZESMALL_IMG_SIZE = (240, 160)

class Image(asl.Type):
  "A rendered image"
  typesize = SMALL_IMG_SIZE

@dispatch(Integer, Integer)
def dist(x, y):
  # Here we are
  import pdb; pdb.set_trace()
  return nn.MSELoss()(x.value, y.value)


word_size = 4   # Size of word
nwords = 10     # Number of terminals in alphabet
nimages = 2     # NUmber of images in set

class Integer(asl.Type):
  "Integer"
  # typesize = (nimages,)
  typesize = SMALL_IMG_SIZE # HACK

class Sentence(asl.Type):
  "A sentence in a language (describing an image)"
  # typesize = (nwords, word_size)
  typesize = SMALL_IMG_SIZE # HACK

class Describe(asl.Function, asl.Net):
  "Describe the ith image of an image set"
  def __init__(self, name="Describe", **kwargs):
    asl.Function.__init__(self, [Image for _ in range(nimages)] + [Integer],
                                [Sentence])
    asl.Net.__init__(self, name, **kwargs)

class WhichImage(asl.Function, asl.Net):
  "Determine which image the sentence describes"
  def __init__(self, name="WhichImage", **kwargs):
    asl.Function.__init__(self, [Image for _ in range(nimages)] + [Sentence],
                                [Integer])
    asl.Net.__init__(self, name, **kwargs)

def train_clevrlang(opt):
  describe = Describe(arch=opt.arch, arch_opt=opt.arch_opt)
  which_image = WhichImage(arch=opt.arch, arch_opt=opt.arch_opt)

  class PermuteTest(asl.Sketch):
    """Generate clevr image from noise"""

    def sketch(self, images,      ):
      # imgj = images[rand_img_id]
      import pdb; pdb.set_trace()
      (sentence, ) = describe(images, rand_img_id)
      permuted_images = random.sample(images, len(images))
      (score, ) = which_image(permuted_images, sentence)
      asl.observe(score, "score")
      return (score, )
    
  def target_score_gen(images, rand_img_id):
    asl.observe(rand_img_id, "score")
    return (rand_img_id, )

  permute_test = PermuteTest([Image for _ in range(nimages)], [Integer])

  # Cuda everything
  asl.cuda(describe, opt.nocuda)
  asl.cuda(which_image, opt.nocuda)

  # Loss
  # CLEVR img generator
  img_dl = clevr_img_dl(opt.batch_size, normalize=False)

  def refresh_clevr(dl):
    "Extract image data and convert tensor to Mnist data type"
    return [asl.refresh_iter(dl, lambda x: Image(x))]

  loss_gen = asl.ref_loss_gen(permute_test,
                              target_score_gen,
                              img_dl,
                              refresh_clevr)

  # Optimization details
  parameters = list(describe.parameters()) + list(which_image.parameters())
  print("LEARNING RATE", opt.lr)
  optimizer = optim.Adam(parameters, lr=opt.lr)
  asl.opt.save_opt(opt)
  if opt.resume_path is not None and opt.resume_path != '':
    asl.load_checkpoint(opt.resume_path, permute_test, optimizer)

  asl.train(loss_gen, optimizer, maxiters=100000,
            cont=asl.converged(1000),
            callbacks=[asl.print_loss(100),
                       #  common.plot_empty,
                       common.plot_observes,
                       common.plot_observes,
                       #  common.plot_internals,
                       asl.save_checkpoint(1000, permute_test)],
            log_dir=opt.log_dir)

if __name__ == "__main__":
  opt = asl.handle_args(clevrlang_args)
  opt = asl.handle_hyper(opt, __file__)
  if opt.sample:
    opt = asl.merge(clevrlang_args_sample(), opt)
  asl.save_opt(opt)
  train_clevrlang(opt)
