"Funcs to search over hyperparameters"
import subprocess
import os
import asl

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

def run_sbatch(file_path, options, sbatch_opt = None, bash_run_path=None):
  """Execute sbatch with options"""
  if bash_run_path is None:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    bash_run_path = os.path.join(dir_path, 'run.sh')

  sbatch_opt = {} if sbatch_opt is None else sbatch_opt
  run_str = ['sbatch'] + make_batch_string(sbatch_opt) + [bash_run_path, file_path] + make_batch_string(options)
  print(run_str)
  subprocess.call(run_str)


# def run_sbatch_chunk(path, chunk, sbatch_opt, batch_run_path=None):
#   for job in chunk:
#     # Save the option file and call sbatch at that location
    
#     asl.save_opt(job)
#     run_sbatch(path, job, sbatc_opt = sbatch_opt,  batch_run_path = batch_run_path)


def run_local_batch(file_path, options, blocking=True):
  """Execute process with options"""
  # opts = mergedict(options, {"--"})
  run_str = ["python", file_path] + make_batch_string(options)
  print("Subprocess call:", run_str)
  # import pdb; pdb.set_trace()
  if blocking:
    subprocess.call(run_str)
  else:
    subprocess.Popen(run_str)


def run_local_chunk(runpath, chunk, blocking=True, batch_run_path=None):
  for job in chunk:
    savefullpath = asl.save_opt(job)
    # Save the option file and call subprocess at that location
    run_local_batch(runpath, {"optfile": savefullpath}, blocking=blocking)

  # TODO: I might want to block at end of each chunk