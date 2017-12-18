# TODO
# 1. Get data in
# 2. GAN loss
# 3.

# In a sense I have a reference for this entire sketch
# But:
# 1. Reference is not really a function, takes no noise
# 2. Not quite true, reference can use noise but its kind of implicit in the
#    minibatch iteraiton\
# 2. not looking for behaviour equivalecne but distributional through means of
# verifier

# Option 1, make a reference gen which just samples from the data
# One problem with the current model is that I want to observe reference
# On the output of the sketch

# I think there are two orthogonal ideas that should be separated
# a) construct a composite function out of other functions, maybe using holes
# in the sketch sense, so it is a sketch

# 2. train with reference

# In this case my reference is just with respect to the sketch as a
# but sometimse we DO have references for the inner nodes

"Learn a generative model of and from Clevr images"
from typing import List
import asl
from asl.modules.modules import ConstantNet, ModuleDict
from torch import optim, nn
import common
from multipledispatch import dispatch

class ClevrGen(asl.Sketch):
  def sketch(self, noise, add_object, render, gen_object, empty_scene):
    """Generate clevr image"""
    # Add object 1
    object1 = gen_object(next(noise))
    scene = empty_scene
    (scene, ) = add_object(scene, object1)

    # Add object 2
    object2 = gen_object(next(noise))
    scene = empty_scene
    (scene, ) = add_object(scene, object2)

    # Add object 3
    object3 = gen_object(next(noise))
    scene = empty_scene
    (scene, ) = add_object(scene, object3)

    (img, ) = render(scene)
    return (img, )

def clevrgen_args(parser):
  parser.add_argument('--batch_norm', action='store_true', default=True,
                      help='Do batch norm')


def clevrgen_args_sample():
  "Options sampler"
  return argparse.Namespace(batch_norm=np.random.rand() > 0.5)

mnist_size = (1, 28, 28)

class Noise(asl.type):
  typesize = (10, 10)

class Scene(asl.Type):
  typesize = (10, 10)

class Object(asl.Type):
  typesize = (10, 10)

CLEVR_IMG_SIZE = (480, 320)
class Image(asl.Type):
  "A rendered image"
  typesize = CLEVR_IMG_SIZE

@dispatch(Image, Image)
def dist(x, y):
  return nn.MSELoss()(x.value, y.value)

def train_clevrgen(**opt):
  arch = opt["arch"]
  arch_opt = opt['']
  class AddObject(asl.Function, asl.Net):
    def __init__(self, name="AddObject", **kwargs):
      asl.Function.__init__(self, [Scene, Object], [Scene])
      asl.Net.__init__(self, name, **kwargs)

  class Render(asl.Function, asl.Net):
    def __init__(self, name="Render", **kwargs):
      asl.Function.__init__(self, [Scene], [Image])
      asl.Net.__init__(self, name, **kwargs)

  class GenObject(asl.Function, asl.Net):
    def __init__(self, name="GenObject", **kwargs):
      asl.Function.__init__(self, [Noise], [Object])
      asl.Net.__init__(self, name, **kwargs)

  nclevrgen = ModuleDict({'add_object': AddObject(arch=opt.arch,
                                    arch_opt=opt.arch_opt),
                          'render': Render(arch=opt.arch,
                                  arch_opt=opt.arch_opt),
                          'gen_object': Render(arch=opt.arch,
                                  arch_opt=opt.arch_opt)})

  clevrgen_sketch = StackSketch([List[Mnist]], [Mnist], nclevrgen, ref_clevrgen())
  asl.cuda(clevrgen_sketch)

  # Loss
  clevr_img_iter = aslbench.clevr.data.ClevrImages(opt.batch_size)
  loss_gen = asl.sketch.loss_gen_gen(clevrgen_sketch,
                                     clevr_img_iter,
                                     lambda x: Mnist(asl.util.data.train_data(x)))

  # Optimization details
  optimizer = optim.Adam(nclevrgen.parameters(), lr=opt.lr)
  asl.opt.save_opt(opt)
  if opt.resume_path is   not None and opt.resume_path != '':
    asl.load_checkpoint(opt.resume_path, nclevrgen, optimizer)

  asl.train(loss_gen, optimizer, maxiters=100000,
        cont=asl.converged(1000),
        callbacks=[asl.print_loss(100),
                   common.plot_empty,
                   common.plot_observes,
                   common.plot_internals,
                   asl.save_checkpoint(1000, nclevrgen)
                   ],
        log_dir=opt.log_dir)


if __name__ == "__main__":
  opt = asl.handle_args(clevrgen_args)
  opt = asl.handle_hyper(opt, __file__)
  if opt.sample:
    opt = asl.merge(clevrgen_args_sample(), opt)
  asl.save_opt(opt)
  train_clevrgen(**vars(opt))
