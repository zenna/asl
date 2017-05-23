"""Training for tensorflow"""
import tensorflow as tf
from pdt.types import Loss
# from wacacore.util.io import *
from wacacore.train.common import train_load_save, updates
# from wacacore.train.callbacks import *


def get_axiom_losses(axioms):
    """Accumulate losses forall axiom in axioms, forall equation in axiom"""
    losses = []
    for i, axiom in enumerate(axioms):
        for j, loss in enumerate(axiom.get_losses()):
            losses.append(Loss(loss, 'ax_%s_%s_%s' % (axiom.name, i, j)))
    return losses


def get_all_losses(adt):
    """Accumulate losses forall axiom in axioms, forall equation in axiom"""
    return get_axiom_losses(adt.axioms) + adt.losses


def get_fetches(adt, options):
    "Create one update functon per loss term"
    losses = get_all_losses(adt)
    losses_dict = {loss.name: loss.loss for loss in losses}
    sum_loss = sum([loss.loss for loss in losses])
    fetch = {}
    fetch['losses'] = losses_dict
    fetch['loss'] = sum_loss
    loss_updates = []
    for l in losses:
        params = []
        if l.restrict_to is not None:
            for i in l.restrict_to:
                params += i.get_params()
        else:
            params = None
        update = updates(l.loss, params, options)[1]
        loss_updates.append(update)
    return fetch, loss_updates


def attach_gen(generators, forallvars):
    while True:
        inputs = [next(gen) for gen in generators]
        feed_dict = {forallvars[i].input_var: inputs[i] for i in range(len(inputs))}
        yield feed_dict


def train(adt,
          pdt,
          options,
          extra_fetches=None,
          callbacks=None):
    """Train the abstract data type"""
    fetch, loss_updates = get_fetches(adt, options)
    fetch['extra_fetches'] = extra_fetches
    sess = tf.Session()
    return train_load_save(sess,
                           loss_updates,
                           fetch,
                           pdt.train_generators,
                           pdt.test_generators,
                           callbacks,
                           options)
