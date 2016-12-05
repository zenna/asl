import matplotlib.pyplot as plt
import numpy as np

def plot_vector(x, **kwargs):
    plt.imshow(x, interpolation='nearest', **kwargs)

def plot_many_vectors(vecs, **kwargs):
    plt.figure()
    nrows = len(vecs)
    ncols = 1
    vec_min = float("inf")
    vec_max = -float("inf")
    for i, vec in enumerate(vecs):
        vec_min = min(np.min(vec), vec_min)
        vec_max = max(np.max(vec), vec_max)
        print(vec_min, vec_max)

    for i, vec in enumerate(vecs):
        plt.subplot(nrows, ncols, i+1)
        plot_vector(vec, vmin=vec_min, vmax=vec_max, **kwargs)
