"""Training for tensorflow"""
from wacacore.util.io import *
from wacacore.train.common import train_loop, get_updates, do_load, do_save, prep_load, prep_save
from wacacore.util.misc import inn, getn
from wacacore.train.callbacks import *
import time
import os
import numpy as np
import tensorflow as tf

def get_losses(axioms):
    """Accumulate losses forall axiom in axioms, forall equation in axiom"""
    losses = {}
    for i, axiom in enumerate(axioms):
        for j, loss in enumerate(axiom.get_losses()):
            losses['ax_%s_%s_%s' % (axiom.name, i, j)] = loss
    return losses

def get_fetches(axioms, options):
    fetch = {}
    losses = get_losses(axioms)
    loss = sum(losses.values())
    fetch['losses'] = losses
    fetch['loss'] = loss
    optimizer, update_step = get_updates(loss, options)
    loss_updates = [update_step]
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
    fetch, loss_updates = get_fetches(adt.axioms, options)
    generators = [the_gen(pdt.generators, adt.forallvars)]
    sess = tf.Session()
    saver = tf.train.Saver()
    options['saver'] = saver
    if do_load(options):
        prep_load(sess, saver, options['params_file'])
    else:
        tf.initialize_all_variables()
    if do_save(options):
        options['savedir'] = prep_save(options['dirname'], options['datadir'])

    callbacks = [save_options, save_every_n, save_everything_last]

    if options['train'] is True:
        train_loop(sess,
                   loss_updates,
                   fetch,
                   generators,
                   test_generators=None,
                   loss_ratios=None,
                   callbacks=callbacks,
                   **options)
    return sess
