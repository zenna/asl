"Packing and Unpacking multiple arguments into one for neural networkp "
import asl
import torch

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


def stretch_cat(xs, out_size, same_dim):
  return cat_channels(vec_stretch(xs, out_size, same_dim))
  # tile it
  # matrix/tensor multiplication

def cat_channels(xs, channel_dim=1):
  return torch.cat(xs, dim=channel_dim)
