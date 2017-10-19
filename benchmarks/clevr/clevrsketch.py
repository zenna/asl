"""Clevr Benchmark """
# FIXME: CLEANUP import hell
import asl.opt
from asl.archs.convnet import ConvNet
from asl.archs.mlp import MLPNet
from asl.callbacks import every_n, print_loss, converged, save_checkpoint
from asl.util.misc import cuda
from asl.train import train, max_iters
from asl.modules.modules import ModuleDict
from asl.log import log_append, log
from asl.loss import vec_dist
from torch import optim

import torch
from typing import Any
from torch import nn
from benchmarks.clevr.clevr import scenes_iter, data_iter, ref_clevr, interpret, ans_tensor
from benchmarks.clevr.defns import *
from asl.sketch import Sketch

class ClevrSketch(Sketch):
  "Sketch for copy of list of elements"

  def __init__(self, model, ref_model):
    super(ClevrSketch, self).__init__(Any, Any, model, ref_model)

  def sketch(self, progs, objsets, rels, **kwargs):
    results = []
    def netapply(fname, inputs):
      f = kwargs[fname]
      return f(*inputs)[0]
    def tensor(value):
      x = value.tensor()
      return x.expand(1, *x.size())
    for (i, _) in enumerate(objsets):
      res = interpret(progs[i], objsets[i], rels[i], apply=netapply,
                      value_transform=tensor)
      results.append(res)

    return results

def accuracy(est, target):
  "Find accuracy of estimation for target"
  est_int = [torch.max(t, 1)[1] for t in est]
  target_int = [torch.max(ans, 0)[1] for ans in target]
  diffs  = [est_int[i].data[0] == target_int[i].data[0] for i in range(len(target))]
  ncorrect = sum(diffs)
  acc = ncorrect / len(diffs)
  return acc, ncorrect

def print_accuracy(every, log_tb=True):
  "Print accuracy per every n"
  def print_accuracy_gen(every):
    running_accuracy = 0.0
    while True:
      data = yield
      acc = data.log['accuracy']
      running_accuracy += acc
      if (data.i + 1) % every == 0:
        accuracy_per_sample = running_accuracy / every
        print('accuracy per sample (avg over %s) : %.3f %%' % (every, accuracy_per_sample))
        if log_tb:
          data.writer.add_scalar('accuracy', accuracy_per_sample, data.i)
        running_accuracy = 0.0
  gen = print_accuracy_gen(every)
  next(gen)
  return gen


def clevr_args(parser):
  parser.add_argument('--share_funcs', action='store_true', default=False,
                      help='Sample parameter values')


def benchmark_clevr_sketch(share_funcs,
                           batch_size,
                           arch,
                           log_dir,
                           lr,
                           arch_opt,
                           sample,
                           **kwargs):
  arch = MLPNet
  arch_opt = MLPNet

  neu_clevr = {'unique': Unique(arch=arch, arch_opt=arch_opt, sample=sample),
               'relate': Relate(arch=arch, arch_opt=arch_opt, sample=sample),
               'count': Count(arch=arch, arch_opt=arch_opt, sample=sample),
               'exist': Exist(arch=arch, arch_opt=arch_opt, sample=sample),
               'intersect': Intersect(arch=arch, arch_opt=arch_opt, sample=sample),
               'union': Union(arch=arch, arch_opt=arch_opt, sample=sample),
               'greater_than': GreaterThan(arch=arch, arch_opt=arch_opt, sample=sample),
               'less_than': LessThan(arch=arch, arch_opt=arch_opt, sample=sample),
               'equal_integer': EqualInteger(arch=arch, arch_opt=arch_opt, sample=sample)}
  if share_funcs:
    fil = Filter(arch=arch, arch_opt=arch_opt, sample=sample)
    eq = Equal(arch=arch, arch_opt=arch_opt, sample=sample)
    query = Query(arch=arch, arch_opt=arch_opt, sample=sample)
    same = Same(arch=arch, arch_opt=arch_opt, sample=sample)
    neu_clevr.update({'filter_size': fil,
                      'filter_color': fil,
                      'filter_material': fil,
                      'filter_shape': fil,
                      'equal_material': eq,
                      'equal_size': eq,
                      'equal_shape': eq,
                      'equal_color': eq,
                      'query_shape': query,
                      'query_size': query,
                      'query_material': query,
                      'query_color': query,
                      'same_shape': same,
                      'same_size': same,
                      'same_material': same,
                      'same_color': same})
  else:
    neu_clevr.update({'filter_size': FilterSize(arch=arch, arch_opt=arch_opt, sample=sample),
                      'filter_color': FilterColor(arch=arch, arch_opt=arch_opt, sample=sample),
                      'filter_material': FilterMaterial(arch=arch, arch_opt=arch_opt, sample=sample),
                      'filter_shape': FilterShape(arch=arch, arch_opt=arch_opt, sample=sample),
                      'equal_material': EqualMaterial(arch=arch, arch_opt=arch_opt, sample=sample),
                      'equal_size': EqualSize(arch=arch, arch_opt=arch_opt, sample=sample),
                      'equal_shape': EqualShape(arch=arch, arch_opt=arch_opt, sample=sample),
                      'equal_color': EqualColor(arch=arch, arch_opt=arch_opt, sample=sample),
                      'query_shape': QueryShape(arch=arch, arch_opt=arch_opt, sample=sample),
                      'query_size': QuerySize(arch=arch, arch_opt=arch_opt, sample=sample),
                      'query_material': QueryMaterial(arch=arch, arch_opt=arch_opt, sample=sample),
                      'query_color': QueryColor(arch=arch, arch_opt=arch_opt, sample=sample),
                      'same_shape': SameShape(arch=arch, arch_opt=arch_opt, sample=sample),
                      'same_size': SameSize(arch=arch, arch_opt=arch_opt, sample=sample),
                      'same_material': SameMaterial(arch=arch, arch_opt=arch_opt, sample=sample),
                      'same_color': SameColor(arch=arch, arch_opt=arch_opt, sample=sample)})

  neuclevr = ModuleDict(neu_clevr)
  refclevr = ref_clevr
  clevr_sketch = ClevrSketch(neuclevr, refclevr)
  cuda(clevr_sketch)
  data_itr = data_iter(batch_size)

  def loss_gen():
    nonlocal data_itr
    try:
      progs, objsets, rels, answers = next(data_itr)
    except StopIteration:
      data_itr = data_iter(batch_size)
      progs, objsets, rels, answers = next(data_itr)

    outputs = clevr_sketch(progs, objsets, rels)
    anstensors = [ans_tensor(ans) for ans in answers]
    acc, ncorrect = accuracy(outputs, anstensors)
    log("accuracy", acc)
    log("ncorrect", ncorrect)
    log("outof", len(answers))
    deltas = [nn.BCEWithLogitsLoss()(outputs[i][0], anstensors[i]) for i in range(len(outputs))]
    return sum(deltas) / len(deltas)


  optimizer = optim.Adam(clevr_sketch.parameters(), lr)
  train(loss_gen,
        optimizer,
        cont=converged(1000),
        callbacks=[print_loss(10),
                  print_accuracy(10),
                  #  every_n(plot_sketch, 500),
                  #  common.plot_empty,
                  #  common.plot_observes,
                   save_checkpoint(1000, clevr_sketch)],
        log_dir=log_dir)

if __name__ == "__main__":
  opt = asl.opt.handle_args(clevr_args)
  opt = asl.opt.handle_hyper(opt, __file__)
  asl.opt.save_opt(opt)
  benchmark_clevr_sketch(**vars(opt))
