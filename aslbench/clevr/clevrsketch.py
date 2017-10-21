"""Clevr Benchmark """
from typing import Any
import argparse
import asl
import asl.archs as archs
import asl.util as util
from torch import optim
import numpy as np
import aslbench
from aslbench.clevr.clevr import data_iter, ref_clevr, interpret, ans_tensor
# import aslbench.clevr as clevr


class ClevrSketch(asl.Sketch):
  "Sketch for copy of list of elements"

  def __init__(self, model, ref_model):
    super(ClevrSketch, self).__init__(Any, Any, model, ref_model)

  def sketch(self, progs, objsets, rels, **kwargs):
    results = []

    def netapply(fname, inputs):
      f = kwargs[fname]
      return f(*inputs)[0]

    def tensor(value):
      return asl.onehot1d(value)

    for (i, _) in enumerate(objsets):
      res = interpret(progs[i], objsets[i], rels[i], apply=netapply,
                      value_transform=tensor)
      results.append(res)

    return results


def accuracy(est, target):
  "Find accuracy of estimation for target"
  eqs = [asl.equal(est[i], target[i]) for i in range(len(est))]
  ncorrect = sum(eqs)
  acc = ncorrect / len(eqs)
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
  parser.add_argument('--batch_norm', action='store_true', default=False,
                      help='Do batch norm')


def clevr_args_sample():
  "Options sampler"
  return argparse.Namespace(share_funcs=np.random.rand() > 0.5,
                            batch_norm=np.random.rand() > 0.5)

def benchmark_clevr_sketch(share_funcs,
                           batch_norm,
                           batch_size,
                           arch,
                           log_dir,
                           lr,
                           arch_opt,
                           sample,
                           **kwargs):
  arch = archs.MLPNet
  sample_args = {'pbatch_norm': int(batch_norm)}
  funs = aslbench.clevr.genfuns.func_types()
  neu_clevr = aslbench.clevr.arch.funcs(arch, arch_opt, sample, sample_args, **funs)
  neuclevr = asl.modules.modules.ModuleDict(neu_clevr)
  refclevr = ref_clevr
  clevr_sketch = ClevrSketch(neuclevr, refclevr)
  util.cuda(clevr_sketch)
  data_itr = data_iter(batch_size)

  {ColorEnum: onehot1d}


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
    asl.log("accuracy", acc)
    asl.log("ncorrect", ncorrect)
    asl.log("outof", len(answers))
    deltas = [asl.dist(outputs[i], anstensors[i]) for i in range(len(outputs))]
    return sum(deltas) / len(deltas)


  optimizer = optim.Adam(clevr_sketch.parameters(), lr)
  asl.train(loss_gen,
            optimizer,
            cont=asl.converged(100),
            callbacks=[asl.print_loss(10),
                       print_accuracy(10),
                       asl.nancancel,
                       #  every_n(plot_sketch, 500),
                       #  common.plot_empty,
                       #  common.plot_observes,
                       asl.save_checkpoint(100, clevr_sketch)],
        log_dir=log_dir)
