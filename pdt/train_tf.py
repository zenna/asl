"""Training for tensorflow"""
from pdt.util.io import *
import time
import os
import numpy as np
import tensorflow as tf
from wacacore.train.common import train_loop, get_updates

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
          options,
          save_dir,
          sfx,
          num_iterations=10000):
    """Train the abstract data type"""
    fetch, loss_updates = get_fetches(adt.axioms, options)
    generators = [the_gen(pdt.generators, adt.forallvars)]

    options_path = os.path.join(save_dir, "options")
    save_dict_csv(options_path, options)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()

    if options['load_params'] is True:
        saver.restore(sess, options['params_file'])

    if options['train'] is True:
        train_loop(sess,
                   loss_updates,
                   fetch,
                   generators,
                   test_generators=None,
                   loss_ratios=None,
                   test_every=100,
                   num_iterations=num_iterations,
                   callbacks=[])

    return sess



#
# def train(adt,
#           pdt,
#           sess,
#           num_epochs=10000,
#           summary_gap=500,
#           save_every=10,
#           sfx='',
#           compress=False,
#           save_dir="./",
#           saver=None):
#     """One epoch is one pass through the data set"""
#     print("Starting training...")
#     j = 0
#     stats = {'loss_vars': [], 'loss_sums': []}
#     for epoch in range(num_epochs):
#         train_err = 0
#         train_batches = 0
#         start_time = time.time()
#         ntrain_outs = len(pdt.train_outs)
#         train_outs = None
#         [next(gen) for gen in pdt.generators]
#         for i in range(summary_gap):
#             gens = [gen.send(train_outs) for gen in pdt.generators]
#             inputs = pdt.gen_to_inputs(gens)
#             assert len(inputs) == len(adt.forallvars)
#             feed_dict = {adt.forallvars[i].input_var: inputs[i] for i in range(len(inputs))}
#             train_outs_losses = pdt.train_fn(feed_dict, sess)
#             train_outs = train_outs_losses[0:ntrain_outs]
#             losses = train_outs_losses[ntrain_outs:]
#             print("epoch:", epoch, " of ", num_epochs, " i: ", i, "losses: ", losses)
#             train_err += losses[-2]
#             train_batches += 1
#             gens = [next(gen) for gen in pdt.generators]
#             if j % save_every == 0:
#                 print(dict(zip([axiom.name for axiom in adt.axioms], losses[0:-1])))
#                 save_path = os.path.join(save_dir, "model.ckpt")
#                 save_path = saver.save(sess, save_path)
#                 print("Model saved in file: %s" % save_path)
#             j = j + 1
#             #     # Savs statistics
#             #     loss_sum = np.sum(losses)
#             #     stats['loss_sums'].append(loss_sum)
#             #     loss_var = np.var(losses)
#             #     stats['loss_vars'].append(loss_var)
#             #     stat_sfx = "epoch_%s_run_%s_stats" % (epoch, i)
#             #     if compress:
#             #         np.savez_compressed(stats_path, **stats)
#             #     else:
#             #         np.savez(stats_path, **stats)
#             #     # Save Params
#             #     sfx2 = "epoch_%s_run_%sloss_%s" % (epoch, i, str(loss_sum))
#             #     path = os.path.join(save_dir, sfx2)
#             #     # adt.save_params(path, compress=compress)
#         print("epoch: ", epoch, " Total loss per epoch: ", train_err)
#     # path = os.path.join(save_dir, "final" + sfx)
#     # adt.save_params(path)
