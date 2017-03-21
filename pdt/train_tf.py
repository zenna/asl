"""Training for tensorflow"""
from wacacore.util.io import *
from wacacore.train.common import train_loop, get_updates, do_load, do_save, prep_load, prep_save
from wacacore.util.misc import inn, getn
from wacacore.train.callbacks import *
import time
import os
import numpy as np
import tensorflow as tf

def get_axiom_losses(axioms):
    """Accumulate losses forall axiom in axioms, forall equation in axiom"""
    losses = {}
    for i, axiom in enumerate(axioms):
        for j, loss in enumerate(axiom.get_losses()):
            losses['ax_%s_%s_%s' % (axiom.name, i, j)] = loss
    return losses

def get_loss_losses(losses):
    ret_losses = {}
    for i, loss in enumerate(losses):
        ret_losses['loss_%s_%s' % (loss.name, i)] = loss.loss
    return ret_losses

def get_all_losses(adt):
    """Accumulate losses forall axiom in axioms, forall equation in axiom"""
    a = get_axiom_losses(adt.axioms)
    a.update(get_loss_losses(adt.losses))
    return a

def get_fetches(adt, options):
    fetch = {}
    losses = get_all_losses(adt)
    loss = sum(losses.values())
    fetch['losses'] = losses
    fetch['loss'] = loss
    # fetch['numerics'] = tf.add_check_numerics_ops()
    # optimizer, update_step = get_updates(loss, options)
    # loss_updates = [update_step]
    loss_updates = [get_updates(loss, options)[1] for loss in losses.values()]
    return fetch, loss_updates


def the_gen(generators, forallvars):
    while True:
        inputs = [next(gen) for gen in generators]
        feed_dict = {forallvars[i].input_var: inputs[i] for i in range(len(inputs))}
        yield feed_dict

def train(adt,
          pdt,
          options):
    """Train the abstract data type"""
    fetch, loss_updates = get_fetches(adt, options)
    train_generators = [the_gen(pdt.train_generators, adt.forallvars)]
    test_generators = [the_gen(pdt.test_generators, adt.forallvars)]
    sess = tf.Session()
    saver = tf.train.Saver()
    options['saver'] = saver
    if do_load(options):
        prep_load(sess, saver, options['params_file'])
    else:
        sess.run(tf.initialize_all_variables())
    if do_save(options):
        options['savedir'] = prep_save(options['dirname'], options['datadir'])
    callbacks = [save_options, save_every_n, save_everything_last, nan_cancel]

    if options['train'] is True:
        train_loop(sess,
                   loss_updates,
                   fetch,
                   train_generators=train_generators,
                   test_generators=test_generators,
                   loss_ratios=None,
                   callbacks=callbacks,
                   **options)
    return sess
