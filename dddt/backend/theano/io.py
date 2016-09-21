import dddt.config._FLOATX

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
