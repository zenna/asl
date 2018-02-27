"Packing and Unpacking multiple arguments into one for neural networkp "
from functools import reduce
import asl
import torch
from torch import nn
from collections import OrderedDict

def nelements(sizes):
  "Total number for elements from set of sizes"
  size = [asl.util.misc.mul_product(size) for size in sizes]
  return sum(size)

def nelem(size):
  "Number for elements from in tensor of size `size`"
  return asl.util.misc.mul_product(size)

def ndims(size):
  "Number of dimensions of size"
  return len(size)

# Unstacking
def split_channel(t, sizes, channel_dim=0, slice_dim=1):
  "Separate t by channel: output[i] takes t[:, 0:sizes[i], :, :] channels"
  assert len(sizes) > 0
  channels = [size[channel_dim] for size in sizes]
  if len(sizes) == 1:
    # print("Only one output skipping unstack")
    return (t,)
  else:
    outputs = []
    c0 = 0
    for c in channels:
      # print("Split ", c0, ":", c0 + c)
      outputs.append(t.narrow(slice_dim, c0, c))
      c0 = c

  return tuple(outputs)

# Unstacking
def split_tensor(t, nelements_, slice_dim=1):
  "Split 2D Array t into several different ones of size nelements_i"
  lb = 0
  outputs = []
  for length in nelements_:
    ub = lb + length
    slice = t.narrow(slice_dim, lb, length)
    outputs.append(slice)
    lb = ub

  return tuple(outputs)


def splt_reshape_tensors(t, sizes):
  nelements_ = [nelem(size) for size in sizes]
  tens = split_tensor(t, nelements_)
  out = [tens[i].contiguous().view(-1, *size) for i, size in enumerate(sizes)]
  return tuple(out)


def gcd(a, b):
    """Return greatest common divisor using Euclid's Algorithm."""
    while b:
        a, b = b, a % b
    return a


def lcm(a, b):
    """Return lowest common multiple."""
    return a * b // gcd(a, b)


def lcmm(*args):
    """Return lcm of args."""
    return reduce(lcm, args)


def make_img_like(img):
  if img.dim() == 4:
    return img
  elif img.dim() == 3:
    return img.view(img.size(0), 1, img.size(1), img.size(2))
  elif img.dim() == 2:
    return img.view(img.size(0), 1, 1, img.size(1))


def lcmimgs(imgs, dim):
  lcmx = lcmm(*[t.size(dim) for t in imgs])
  return lcmx

import torch.nn as nn

def resize(img, lcmx, lcmy, mode='bilinear'):
  scale_factor = (lcmx // img.size(2), lcmy // img.size(3))
  if scale_factor == (1, 1):
    return img
  else:
    return nn.Upsample(scale_factor=scale_factor, mode=mode)(img)


def stackable_imgs(tensors):
  """Convert tensors tensors into images that can be stacked (same width and height)"""
  imgs = list(map(make_img_like, tensors))
  # import pdb; pdb.set_trace()
  lcmx, lcmy = lcmimgs(imgs, 2), lcmimgs(imgs, 3)
  return [resize(img, lcmx, lcmy) for img in imgs]


def slither(x, target_size):
  "Change an image into a slice of it and leave an image untouched"
  if len(target_size) == 1:
    return x[:, 0, :, 0]
  elif len(target_size) == 3:
    return x
  else:
    raise ValueError


def vec_stretch(args, img_size, same_dim):
  """"Arguments
  args: a vector of tensors of dimension 2, or 4
  out_size: dimension of output
  same_dim: dimension of
  """
  target_shape = [1, 1, 1]
  target_shape[same_dim] = img_size[same_dim]
  out_size = [1, img_size[1], img_size[2]]

  def expand(x):
    if x.dim() == 2:
      return x.view(x.size(0), *target_shape).expand(x.size(0), *out_size)
    elif x.dim() == 4:
      return x
    else:
      raise ValueError

  return list(map(expand, args))