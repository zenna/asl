from __future__ import print_function
import numpy as np
import sys
import getopt
import os
import scipy.ndimage
import csv
import time

## Primitive Functions
## ------------------
def upper_div(x, y):
    a, b = divmod(x, y)
    if b == 0:
        return a
    else:
        return a + 1

def identity(x):
    return x


def upper_div(x, y):
    a, b = divmod(x, y)
    if b == 0:
        return a
    else:
        return a + 1



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


# Minibatching
def infinite_samples(sampler, batchsize, shape):
    while True:
        to_sample_shape = (batchsize,)+shape
        yield lasagne.utils.floatX(sampler(*to_sample_shape))


def infinite_batches(inputs, batchsize, f=identity, shuffle=False):
    start_idx = 0
    nelements = len(inputs)
    indices = np.arange(nelements)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    while True:
        end_idx = start_idx + batchsize
        if end_idx > nelements:
            diff = end_idx - nelements
            excerpt = np.concatenate([indices[start_idx:nelements], indices[0:diff]])
            start_idx = diff
            if shuffle:
                indices = np.arange(len(inputs))
                np.random.shuffle(indices)
        else:
            excerpt = indices[start_idx:start_idx + batchsize]
            start_idx = start_idx + batchsize
        yield f(inputs[excerpt])

#
# def infinite_batches(inputs, batchsize, f, shuffle=False):
#     start_idx = 0
#     nelements = len(inputs)
#     indices = np.arange(nelements)
#     if shuffle:
#         indices = np.arange(len(inputs))
#         np.random.shuffle(indices)
#     while True:
#         data = yield
#         end_idx = start_idx + batchsize
#         if end_idx > nelements:
#             diff = end_idx - nelements
#             excerpt = np.concatenate([indices[start_idx:nelements],
#                                       indices[0:diff]])
#             start_idx = diff
#             if shuffle:
#                 indices = np.arange(len(inputs))
#                 np.random.shuffle(indices)
#         else:
#             excerpt = indices[start_idx:start_idx + batchsize]
#             start_idx = start_idx + batchsize
#         yield f(inputs[excerpt], data)


def constant_batches(x, f):
    while True:
        data = yield
        yield f(x, data)


def iterate_batches(inputs, batchsize, shuffle=False):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_iiteratedx + batchsize)
        yield inputs[excerpt]


# def repeat_to_batch(x, batch_size, tnp):
#     return tnp.repeat(x, batch_size, axis=0)

def repeat_to_batch(x, batch_size):
    """
    Tile the first dimension to a batch_size
    x :: tf.Tensor (1, d2, d3, ..., dn) -> tf.Tensor (batch_size, d2, d3, ..., dn)
    """
    import tensorflow as tf
    shape = tf.shape(x)
    rnk = tf.rank(x)
    tileshp = tf.ones([rnk - 1], dtype=tf.int32)
    tileshpfinal = tf.concat(0, [[batch_size], tileshp])
    return tf.tile(x, tileshpfinal)

# Dict Functions
def stringy(ls):
    out = ""
    for l in ls:
        out = out + str(l) + "_"
    return out


def stringy_dict(d):
    out = ""
    for (key, val) in d.items():
        if val is not None and val is not '':
            out = out + "%s_%s__" % (str(key), str(val))
    return out


def save_params(fname, params):
    f = open(fname, 'w')
    writer = csv.writer(f)
    for key, value in params.items():
        writer.writerow([key, value])
    f.close()

def save_dict_csv(fname, params):
    f = open(fname, 'w')
    writer = csv.writer(f)
    for key, value in params.items():
        writer.writerow([str(key), str(value)])
    f.close()

def npz_to_array(npzfile):
    """"Get a list of numpy arrays from a npz file"""
    nitems = len(npzfile.keys())
    return [npzfile['arr_%s' % i]  for i in range(nitems)]

def get_filepaths(directory):
    """
    This function will generate the file names in a directory
    tree by walking the tree either top-down or bottom-up. For each
    directlds a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.


def handle_args(argv, cust_opts):
    custom_long_opts = ["%s=" % k for k in cust_opts.keys()]
    cust_double_dash = ["--%s" % k for k in cust_opts.keys()]
    long_opts = ["params_file=", "learning_rate=", "momentum=", "update=",
                 "description=", "template="] + custom_long_opts
    options = {'params_file': '', 'learning_rate': 0.1, 'momentum': 0.9,
               'load_params': False, 'update': 'momentum', 'description': '',
               'template': 'res_net'}
    help_msg = """-p <paramfile> -l <learning_rate> -m <momentum> -u <update algorithm> -d
                  <job description> -t <template>"""
    try:
        opts, args = getopt.getopt(argv, "hp:l:m:u:d:t:", long_opts)
    except getopt.GetoptError:
        print("invalid options")
        print(help_msg)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(help_msg)
            sys.exit()
        elif opt in ("-p", "--params_file"):
            options['params_file'] = arg
            options['load_params'] = True
        elif opt in ("-l", "--learning_rate"):
            options['learning_rate'] = float(arg)
        elif opt in ("-m", "--momentum"):
            options['momentum'] = float(arg)
        elif opt in ("-u", "--update"):
            if arg in ['momentum', 'adam', 'rmsprop']:
                options['update'] = arg
            else:
                print("update must be in ", ['momentum', 'adam', 'rmsprop'])
                print(help_msg)
                sys.exit()
        elif opt in ("-d", "--description"):
            options['description'] = arg
        elif opt in ("-t", "--template"):
            options['template'] = arg
        elif opt in cust_double_dash:
            opt_key = opt[2:]  # remove --
            cust = cust_opts[opt_key]
            if len(cust) == 1:
                options[opt_key] = True
            elif len(cust) == 2:
                f, default = cust
                options[opt_key] = f(arg)
            else:
                sys.exit()

    # add defaults back
    for (key, val) in cust_opts.items():
        if key not in options:
            options[key] = val[-1]

    print(options)
    return options


def mk_dir(sfx=''):
    "Create directory with timestamp"
    datadir = os.environ['DATADIR']
    newdirname = str(time.time()) + sfx
    full_dir_name = os.path.join(datadir, newdirname)
    print("Data will be saved to", full_dir_name)
    os.mkdir(full_dir_name)
    return full_dir_name
