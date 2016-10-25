# Probabilistic Data Types

pdt is a framework for synthesizing approximate data structures from specification.

### Usage

To synthesize an stack (of mnist digits) from specification:
```
ipython -- pdt/examples/stack.py --template=res_net --nblocks=1 --block_size=1 -u adam -l 0.01 --nitems=2 --batch_size=256 --train 1
```
