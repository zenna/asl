"Funcs to search over hyperparameters"
import subprocess
import os
import asl
import torch.nn.functional as F

SHORTNAMES = {F.elu: "elu",
              F.relu: "relu",
              "activation": "act",
              "batch_size": "bs",
              "dataset": "data",
              "omniglot": "om",
              "mnist": "mn",
              "learn_batch_norm": "lbn",
              "nchannels": "nchan",
              "tracegen": "tg",
              # asl.archs.convnet.Convnet:"convnet",
             }

def stringifyfilename(k, v):
  """Turn a key value into command line argument"""
  k = SHORTNAMES[k] if k in SHORTNAMES else k
  v = SHORTNAMES[v] if v in SHORTNAMES else v
  return "__%s_%s" % (k, v)


def linearizeopt(opt, keys):
  parts = [stringifyfilename(k, opt[k]) for k in keys]
  return "_".join(parts)

def linearizeoptrecur_(opt, keys):
  parts = []
  for k, v in opt.items():
    if k in keys:
      parts.append(stringifyfilename(k, v))
    elif asl.opt.isopt(v):
      parts = parts + linearizeoptrecur_(v, keys)

  return parts


def linearizeoptrecur(opt, keys):
  parts = linearizeoptrecur_(opt, keys)
  return "_".join([str(part) for part in parts])


def stringify(k, v):
  """Turn a key value into command line argument"""
  if v is True:
    return "--%s" % k
  elif v is False:
    return ""
  else:
    return "--%s=%s" % (k, v)


def make_batch_string(options):
  """Turn options into a string that can be passed on command line"""
  batch_string = [stringify(k, v) for k, v in options.items()]
  return batch_string

# Using Slurm #

def maybedryrun(dryrun, run_str, f, *args):
  if dryrun:
    print("Dry Running: ", run_str)
    return None
  else:
    print("Real Running: ", run_str)
    return f(*args)

def run_sbatch(file_path, options, sbatch_opt = None, bash_run_path=None, dryrun=False):
  """Execute sbatch with options"""
  if bash_run_path is None:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    bash_run_path = os.path.join(dir_path, 'run.sh')

  sbatch_opt = {} if sbatch_opt is None else sbatch_opt
  run_str = ['sbatch'] + make_batch_string(sbatch_opt) + [bash_run_path, file_path] + make_batch_string(options)
  maybedryrun(dryrun, run_str, subprocess.call, run_str)


def run_local_batch(file_path, options, blocking=True, dryrun=False):
  """Execute process with options"""
  # opts = mergedict(options, {"--"})
  run_str = ["python", file_path] + make_batch_string(options)
  if blocking:
    maybedryrun(dryrun, run_str, subprocess.call, run_str)
  else:
    maybedryrun(dryrun, run_str, subprocess.Popen, run_str)

def run_local_chunk(runpath, chunk, blocking=True, dryrun=False):
  for job in chunk:
    job["log_dir"] = asl.util.io.log_dir(group=job["group"], comment=job["name"])
    savefullpath = maybedryrun(dryrun, "Save opts", asl.save_opt, job)
    # Save the option file and call subprocess at that location
    print(job)
    run_local_batch(runpath, {"optfile": savefullpath}, blocking=blocking,
                     dryrun=dryrun)

  # TODO: I might want to block at end of each chunk


def run_sbatch_chunk(path, chunk, bash_run_path=None, dryrun=False):
  print("Jobs in chunk: {}".format(len(chunk)))
  for job in chunk:
    id = asl.util.io.id_gen()
    job["log_dir"] = asl.util.io.log_dir(id=id, group=job["group"], comment=job["name"])
    # savefullpath = asl.save_opt(job)
    print(job)
    job_name = "{}_{}".format(id, job["name"])
    job["name"] = job_name
    savefullpath = maybedryrun(dryrun, "Save opts", asl.save_opt, job)
    slurmout = os.path.join(job["log_dir"], "slurm.out")
    sbatch_opt = {'job-name': job_name, 'time': 720, "output": slurmout}
    run_sbatch(path, {"optfile": savefullpath}, sbatch_opt=sbatch_opt,
               bash_run_path=bash_run_path, dryrun=dryrun)