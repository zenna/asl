# Data Driven Data Structures

pdt is a framework for synthesizing approximate data structures from specification.

There is currently a Python (theano) and Lua (Torch) implementation

### Usage

Python stack example:
```
ipython -- pdt/examples/stack.py --template=conv_net --nblocks=1 --block_size=1 -u adam -l 0.01 --nitems=2
```

Torch stack example:

```
th torch/examples/stack.lua -nitems 2 -batch_size 512 -learning_rate 0.0001 -num_epochs 100000

```
