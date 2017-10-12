"""Generators"""
import numpy as np


def infinite_samples(sampler, batch_size, shape, add_batch=False):
  "Generator which samples from sampler distribution forever"
  if add_batch:
    shape = (batch_size, ) + shape

  while True:
    yield sampler(*shape)


def infinite_batches(inputs, batch_size, f=lambda x: x, shuffle=False):
  """Generator which without termintation yields batch_size chunk
  of inputs
  Args:
    inputs:
    batch_size:
    f: arbitrary function to apply to batch
    Shuffle: If True randomly shuffles ordering
  """
  start_idx = 0
  nelements = len(inputs)
  indices = np.arange(nelements)
  if batch_size > nelements:
    reps = batch_size / nelements
    indices = np.tile(indices, int(np.ceil(reps)))[0:batch_size]
  if shuffle:
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
  while True:
    end_idx = start_idx + batch_size
    if end_idx > nelements:
      diff = end_idx - nelements
      excerpt = np.concatenate([indices[start_idx:nelements], indices[0:diff]])
      start_idx = diff
      if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    else:
      excerpt = indices[start_idx:start_idx + batch_size]
      start_idx = start_idx + batch_size
    yield f(inputs[excerpt])


def constant_batches(x, f):
  while True:
    data = yield
    yield f(x, data)
