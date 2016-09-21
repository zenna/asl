def variable(value, dtype, name=None):
    '''Instantiate a tensor variable.
    '''
    if hasattr(value, 'tocoo'):
        _assert_sparse_module()
        return th_sparse_module.as_sparse_variable(value)
    else:
        value = np.asarray(value, dtype=dtype)
        return theano.shared(value=value, name=name, strict=False)


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
