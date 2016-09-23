import numpy as np
import tensorflow as tf
from dddt.config import floatX

def placeholder(shape=None, ndim=None, dtype=floatX, sparse=False, name=None):
    '''Instantiates a placeholder.
    # Arguments
        shape: shape of the placeholder
            (integer tuple, may include None entries).
        ndim: number of axes of the tensor.
            At least one of {`shape`, `ndim`} must be specified.
            If both are specified, `shape` is used.
        dtype: placeholder type.
        name: optional name string for the placeholder.
    # Returns
        Placeholder tensor instance.
    '''
    if not shape:
        if ndim:
            shape = tuple([None for _ in range(ndim)])
    if sparse:
        tf_shape = tf.constant(np.array(list([0 for _ in range(len(shape))]), dtype=np.int64))
        x = tf.sparse_placeholder(dtype, shape=tf_shape, name=name)
    else:
        x = tf.placeholder(dtype, shape=shape, name=name)
    return x


def variable(value, dtype, name=None, broadcastable=None):
    """Instantiate a tensor variable."""
    return tf.Variable(value, dtype=dtype, name=name)


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
