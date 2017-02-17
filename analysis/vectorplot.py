import matplotlib.pyplot as plt
import numpy as np

def plot_vector(x, ylabel=None, **kwargs):
    im = plt.imshow(x, interpolation='nearest', **kwargs)
    plt.axis('off')
    plt.ylabel(ylabel)
    return im

def plot_many_vectors(vecs, **kwargs):
    nrows = len(vecs)
    ncols = 1
    vec_min = float("inf")
    vec_max = -float("inf")
    for i, vec in enumerate(vecs):
        vec_min = min(np.min(vec), vec_min)
        vec_max = max(np.max(vec), vec_max)
        print(vec_min, vec_max)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    # for i, vec in enumerate(vecs):
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(vecs[i], interpolation='nearest', vmin=vec_min, vmax=vec_max, **kwargs)
        print("ARAR")
        ax.axis('off')
    fig.colorbar(im, ax=axes.ravel().tolist())
