"Learn a generative model of and from Clevr images"
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

# How to construct element of Sentence
# - Just output element of domain
# - By combinators
# - Output production paths for grammar
# - Output String (sequence of symbols) (and check adherance to grammar)

# How to construct string
# Seems like this is the same problem, could ouput elements of the wrong alphabet

# How to check grammaticity
#

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


class Probability(asl.Type):
  "A number between 0 and 1"
  typesize = (1,)

class Image(asl.Type):
  "A rendered image"
  typesize = SMALL_IMG_SIZE

class ImageSet(asl.Type):
  "A rendered image"
  typesize = SMALL_IMG_SIZE

word_size = 4   # Size of word
nwords = 10     # Number of terminals in alphabet
nimages = 3     # NUmber of images in set

class Integer(asl.Type):
  "Integer"
  typesize = (nimages,)

class Sentence(asl.Type):
  "A rendered image"
  typesize = (nwords, word_size)

class Describe(asl.Function, asl.Net):
  "Add object to scene"
  def __init__(self, name="Describe", **kwargs):
    asl.Function.__init__(self, [ImageSet], [Sentence])
    asl.Net.__init__(self, name, **kwargs)

class WhichImage(asl.Function, asl.Net):
  "WhichImage a scene to an image"
  def __init__(self, name="WhichImage", **kwargs):
    asl.Function.__init__(self, [ImageSet], [Integer])
    asl.Net.__init__(self, name, **kwargs)


def train_clevrlang(opt):
  describe = Describe(arch=opt.arch, arch_opt=opt.arch_opt)
  which_image = WhichImage(arch=opt.arch, arch_opt=opt.arch_opt)

  class PermuteTest(asl.Sketch):
    """Generate clevr image from noise"""

    def sketch(self, images, rand_img_ids):
      imgj = images[rand_img_ids]
      (l, ) = describe(images, rand_img_ids)
      (score, ) = which_image(images, l)
      return (score, rand_img_ids)

  permute_test = PermuteTest([ImageSet], [Image])

  # Cuda everything
  asl.cuda(nclevrlang, opt.nocuda)
  asl.cuda(clevrlang_sketch, opt.nocuda)

  # Loss
  img_dl = clevr_img_dl(opt.batch_size, normalize=False)
  loss_gen = asl.ref_loss_gen(clevrlang_sketch,
                              ref_img_gen,
                              img_dl,
                              lambda x: Image(asl.cuda(Variable(x), opt.nocuda)))

  # Optimization details
  parameters = nclevrlang.parameters()
  optimizer = optim.Adam(parameters, lr=opt.lr)
  asl.opt.save_opt(opt)
  if opt.resume_path is not None and opt.resume_path != '':
    asl.load_checkpoint(opt.resume_path, nclevrlang, optimizer)

  asl.train(loss_gen, optimizer, maxiters=100000,
            cont=asl.converged(1000),
            callbacks=[asl.print_loss(100),
                       #  common.plot_empty,
                       common.log_observes,
                       common.plot_observes,
                       #  common.plot_internals,
                       asl.save_checkpoint(1000, nclevrlang)],
            log_dir=opt.log_dir)


if __name__ == "__main__":
  opt = asl.handle_args(clevrlang_args)
  opt = asl.handle_hyper(opt, __file__)
  if opt.sample:
    opt = asl.merge(clevrlang_args_sample(), opt)
  asl.save_opt(opt)
  train_clevrlang(opt)
