"""Training for tensorflow"""
from wacacore.util.io import *
import time
import os
import numpy as np
import tensorflow as tf
from wacacore.train.common import train_loop, get_updates, prep_save
from wacacore.util.misc import inn, getn
from wacacore.train.callbacks import *

def get_losses(axioms):
    """Accumulate losses forall axiom in axioms, forall equation in axiom"""
    losses = {}
    for i, axiom in enumerate(axioms):
        for j, loss in enumerate(axiom.get_losses()):
            losses['ax_%s_%s_%s' % (axiom.name, i, j)] = loss
    return losses

def get_fetches(axioms, options):
    print("Compiling training fn...")
    fetch = {}
    losses = get_losses(axioms)
    loss = sum(losses.values())
    fetch['losses'] = losses
    fetch['loss'] = loss
    optimizer, update_step = get_updates(loss, options)
    loss_updates = [update_step]
    return fetch, loss_updates


def the_gen(generators, forallvars):
    # Do this first so that its ready for data
    # [next(gen) for gen in pdt.generators]
    while True:
        inputs = [next(gen) for gen in generators]
        # gens = [gen.send(train_outs) for gen in pdt.generators]
        # train_outs = train_outs_losses[0:ntrain_outs]
        feed_dict = {forallvars[i].input_var: inputs[i] for i in range(len(inputs))}
        yield feed_dict

def train(adt,
          pdt,
          options):
    """Train the abstract data type"""
    fetch, loss_updates = get_fetches(adt.axioms, options)
    generators = [the_gen(pdt.generators, adt.forallvars)]
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    if inn(options, 'save', 'dirname', 'params_file', 'datadir', 'load'):
        ops = prep_save(sess, *getn(options, 'save', 'dirname', 'params_file', 'datadir', 'load'))
        options.update(ops)

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
