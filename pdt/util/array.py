import numpy as np


def circular_indices(lb, ub, thresh):
    indices = []
    while True:
        stop = min(ub, thresh)
        ix = np.arange(lb, stop)
        indices.append(ix)
        if stop != ub:
            diff = ub - stop
            lb = 0
            ub = diff
        else:
            break

    return np.concatenate(indices)


def npz_to_array(npzfile):
    """"Get a list of numpy arrays from a npz file"""
    nitems = len(npzfile.keys())
    return [npzfile['arr_%s' % i]  for i in range(nitems)]
