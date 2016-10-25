import tensorflow as tf

def bound_loss(x, a, b):
    """Euclidean distance to interval [a, b]"""
    a = tf.constant(a, shape=x.get_shape())
    b = tf.constant(b, shape=x.get_shape())
    zero = tf.constant(0.0, shape=x.get_shape())
    print((x>b).get_shape())
    print("s@",(x-b).get_shape())
    g = tf.select(x>b, x-b, tf.select(x<a, a-x, zero))
    return tf.reduce_mean(g,reduction_indices=dims_bar_batch(g))

def mse(a, b):
    """Mean square error"""
    eps = 1e-9
    return tf.reduce_mean(tf.square(a - b))

def mae(a, b):
    """Mean absolute error"""
    eps = 1e-9
    return tf.reduce_mean(tf.abs(a - b))
