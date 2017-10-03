# Algebraic Structure Learning


[![Build Status](https://travis-ci.org/zenna/asl.svg?branch=master)](https://travis-ci.org/zenna/asl)

asl is a framework for synthesizing approximate data structures

### Usage

To synthesize an stack (of mnist digits) from specification:
```
ipython -- examples/stack.py --template=res_net --nblocks=1 --block_size=1 -u adam -l 0.0001 --nitems=1 --batch_size=128 --train 1 --num_epochs=1000
```
