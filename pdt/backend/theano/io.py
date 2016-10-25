def placeholder(shape=None, ndim=None, dtype=_FLOATX, sparse=False, name=None):
    '''Instantiate an input data placeholder variable.
    '''
    if shape is None and ndim is None:
        raise Exception('Specify either a shape or ndim value.')
    if shape is not None:
        ndim = len(shape)
    else:
        shape = tuple([None for _ in range(ndim)])

    broadcast = (False,) * ndim
    if sparse:
        _assert_sparse_module()
        x = th_sparse_module.csr_matrix(name=name, dtype=dtype)
    else:
        x = T.TensorType(dtype, broadcast)(name)
    x._keras_shape = shape
    x._uses_learning_phase = False
    return x


def variable(value, dtype=_FLOATX, name=None, **kwargs):
    '''Instantiate a tensor variable.
    '''
    if hasattr(value, 'tocoo'):
        _assert_sparse_module()
        return th_sparse_module.as_sparse_variable(value)
    else:
        value = np.asarray(value, dtype=dtype)
        return theano.shared(value=value, name=name, strict=False, **kwargs)

def repeat_to_batch(x, batch_size, tnp):
    return tnp.repeat(x, batch_size, axis=0)
