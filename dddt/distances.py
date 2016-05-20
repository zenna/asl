import theano.tensor as T


def mse(a, b, tnp=T):
    eps = 1e-9
    return (tnp.maximum(eps, (a - b)**2)).mean()


def absdiff(a, b, tnp=T):
    eps = 1e-9
    return (tnp.maximum(eps, T.abs_(a - b))).mean()


def bound_loss(x, tnp=T):
    eps = 1e-9
    loss = tnp.maximum(tnp.maximum(eps,x-1), tnp.maximum(eps,-x)) + eps
    return tnp.maximum(loss, eps) + eps
