"""Training for tensorflow"""
from pdt.types import Loss
from wacacore.util.io import *
from wacacore.train.common import *
from wacacore.util.misc import inn, getn
from wacacore.train.callbacks import *
import time
import os
import numpy as np
import tensorflow as tf


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
    if 'debug' in options and options['debug'] is True:
        fetch['numerics'] = tf.add_check_numerics_ops()

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

    # Saving and Loading
    saver = tf.train.Saver()
    options['saver'] = saver
    if do_load(options):
        prep_load(sess, saver, options['params_file'])
    else:
        init = tf.global_variables_initializer()
        sess.run(init)
    if do_save(options):
        options['savedir'] = prep_save(options['dirname'], options['datadir'])

    # Summaries
    summaries_dir = os.path.join(options['savedir'], "summaries")
    summaries = variable_summaries(fetch['losses'])
    fetch['summaries'] = summaries
    writers = setup_file_writers(options['savedir'], sess)
    options['writers'] = writers
    #
    # callbacks = [save_options,
    #              save_every_n,
    #              save_everything_last,
    #              nan_cancel,
    #              summary_writes]

    callbacks = [every_n(summary_writes, 10)]

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
