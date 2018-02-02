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


def clevrgen_args(parser):
  "Commandline arguments for clevrgen"
  parser.add_argument('--batch_norm', action='store_true', default=True,
                      help='Do batch norm')


def clevrgen_args_sample():
  "Options sampler"
  return argparse.Namespace(batch_norm=np.random.rand() > 0.5)


CLEVR_IMG_SIZE = (3, 80, 120)
SMALL_IMG_SIZE = CLEVR_IMG_SIZE


class Noise(asl.Type):
  "Noise"
  typesize = SMALL_IMG_SIZE


class Scene(asl.Type):
  "A collection of objects"
  typesize = SMALL_IMG_SIZE


class Object(asl.Type):
  "A geometric object"
  typesize = SMALL_IMG_SIZE


class Image(asl.Type):
  "A rendered image"
  typesize = SMALL_IMG_SIZE


class Probability(asl.Type):
  "A number between 0 and 1"
  typesize = (1,)


@dispatch(Image, Image)
def dist(x, y):
  return nn.MSELoss()(x.value, y.value)


def train_clevrgen(opt):
  class AddObject(asl.Function, asl.Net):
    "Add object to scene"
    def __init__(self, name="AddObject", **kwargs):
      asl.Function.__init__(self, [Scene, Object], [Scene])
      asl.Net.__init__(self, name, **kwargs)

  class Render(asl.Function, asl.Net):
    "Render a scene to an image"
    def __init__(self, name="Render", **kwargs):
      asl.Function.__init__(self, [Scene], [Image])
      asl.Net.__init__(self, name, **kwargs)

  class GenObject(asl.Function, asl.Net):
    "Sample an object from noise"
    def __init__(self, name="GenObject", **kwargs):
      asl.Function.__init__(self, [Noise], [Object])
      asl.Net.__init__(self, name, **kwargs)

  nclevrgen = ModuleDict({'add_object': AddObject(arch=opt.arch,
                                                  arch_opt=opt.arch_opt),
                          'render': Render(arch=opt.arch,
                                           arch_opt=opt.arch_opt),
                          'gen_object': Render(arch=opt.arch,
                                               arch_opt=opt.arch_opt),
                          'empty_scene': ConstantNet(Scene)})

  class Discriminate(asl.Function, asl.Net):
    "Is ``Image`` from data distribution?"
    def __init__(self, name="Discriminate", **kwargs):
      asl.Function.__init__(self, [Image], [Probability])
      asl.Net.__init__(self, name, **kwargs)

  discriminate = Discriminate(arch=opt.arch, arch_opt=opt.arch_opt)

  class ClevrGen(asl.Sketch):
    """Generate clevr image from noise"""

    def sketch(self, noise):
      """Generate clevr image"""
      # Add object 1
      #FIXME this is a hack
      noisetensor = asl.cuda(Variable(torch.rand((1,) + SMALL_IMG_SIZE)), opt.nocuda)
      nnoise = Noise(expand_to_batch(noisetensor, opt.batch_size))
      (object1, ) = nclevrgen.gen_object(nnoise)
      scene = nclevrgen.empty_scene
      (scene, ) = nclevrgen.add_object(scene, object1)

      # Add object 2
      (object2, ) = nclevrgen.gen_object(nnoise)
      (scene, ) = nclevrgen.add_object(scene, object2)

      # Add object 3
      (object3, ) = nclevrgen.gen_object(nnoise)
      (scene, ) = nclevrgen.add_object(scene, object3)

      (img, ) = nclevrgen.render(scene)
      asl.observe(img, 'rendered_img')
      return (img, )

  def ref_img_gen(img_iter):
    img = next(img_iter)
    asl.observe(img, 'rendered_img')
    return (img, )

  @dispatch(Image, Image)
  def dist(x, y):
    return discriminate(x.value, y.value)

  clevrgen_sketch = ClevrGen([Noise], [Image])

  # Cuda everything
  asl.cuda(nclevrgen, opt.nocuda)
  asl.cuda(clevrgen_sketch, opt.nocuda)

  # Loss
  img_dl = clevr_img_dl(opt.batch_size, normalize=False)
  loss_gen = asl.ref_loss_gen(clevrgen_sketch,
                              ref_img_gen,
                              img_dl,
                              lambda ten: Image(asl.cuda(Variable(ten),
                                                opt.nocuda)))

  # Optimization details
  parameters = nclevrgen.parameters()
  optimizer = optim.Adam(parameters, lr=opt.lr)
  asl.opt.save_opt(opt)
  if opt.resume_path is not None and opt.resume_path != '':
    asl.load_checkpoint(opt.resume_path, nclevrgen, optimizer)

  asl.train(loss_gen, optimizer, maxiters=100000,
            cont=asl.converged(1000),
            callbacks=[asl.print_loss(100),
                       #  common.plot_empty,
                       common.log_observes,
                       common.plot_observes,
                       #  common.plot_internals,
                       asl.save_checkpoint(1000, nclevrgen)],
            log_dir=opt.log_dir)


if __name__ == "__main__":
  opt = asl.handle_args(clevrgen_args)
  opt = asl.handle_hyper(opt, __file__)
  if opt.sample:
    opt = asl.merge(clevrgen_args_sample(), opt)
  asl.save_opt(opt)
  train_clevrgen(opt)
