"Helpers for running on OpenMind cluster"
import subprocess
import itertools

def run_batch_scripts():
  nitems = [2, 5]
  #scripts = ['stack.py', 'stackEqualityAxiom.py', 'queue_jx.py', 'queueEqualityAxiom.py']
  scripts = ['queueEqualityAxiom.py']
  l = [nitems, scripts]

  for i in itertools.product(nitems, scripts):
    print(i)
    script = 'sbatch --gres=gpu:1 --mem=16000 -n 4 ipython -- examples/%s --template=conv_res_net --nblocks=1 --block_size=1 -u adam -l 0.0001 --nitems=%s --batch_size=128 --train 1 --num_epochs=1000' % (i[1], i[0])
    subprocess.call(script.split())


if __name__=="__main__":
  run_batch_scripts()
