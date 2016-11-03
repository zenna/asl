"""Training for tensorflow"""
import time
import os
import numpy as np
import tensorflow as tf


def get_updates(loss, options):

    if options['update'] == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=options['learning_rate'],
                                               momentum=options['momentum'])
    elif options['update'] == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=options['learning_rate'])
    elif options['update'] == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learning_rate=options['learning_rate'])
    else:
        assert False, "Unknown loss minimizer"
    update_step = optimizer.minimize(loss)
    return optimizer, update_step


def get_losses(axioms):
    """Accumulate losses forall axiom in axioms, forall equation in axiom"""
    losses = []
    for axiom in axioms:
        for loss in axiom.get_losses():
            losses.append(loss)
    return losses


def get_params(funcs, options, **tags):
    """Accumulate losses forall axiom in axioms, forall equation in axiom"""
    params = []
    for func in funcs:
        for param in func.get_params(**tags):
            params.append(param)

    return params


def compile_fns(funcs, consts, forallvars, axioms, train_outs, options):
    print("Compiling training fn...")
    with tf.name_scope('losses'):
        losses = get_losses(axioms)
        loss = sum(losses)
    # func_params = get_params(funcs, options, trainable=True)
    # constant_params = get_params(consts, options)
    # params = func_params + constant_params
    outputs = train_outs + losses + [loss]
    optimizer, update_step = get_updates(loss, options)
    outputs.append(update_step)

    def train_fn(feed_dict, sess):
        return sess.run(outputs, feed_dict=feed_dict)

    # train_fn = function([forallvar.input_var for forallvar in forallvars],
    #                     outputs, updates=updates)
    # Compile the func for use
    # if options['compile_fns']:
    #     print("Compiling func fns...")
    #     call_fns = [func.compile() for func in funcs]
    # else:
    #     call_fns = []
    # # FIXME Trainable=true, deterministic = true/false
    call_fns = []
    return train_fn, call_fns


def train(adt,
          pdt,
          sess,
          num_epochs=10000,
          summary_gap=500,
          save_every=10,
          sfx='',
          compress=False,
          save_dir="./",
          saver=None):
    """One epoch is one pass through the data set"""
    print("Starting training...")
    j = 0
    stats = {'loss_vars': [], 'loss_sums': []}
    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        ntrain_outs = len(pdt.train_outs)
        train_outs = None
        [next(gen) for gen in pdt.generators]
        for i in range(summary_gap):
            gens = [gen.send(train_outs) for gen in pdt.generators]
            inputs = pdt.gen_to_inputs(gens)
            assert len(inputs) == len(adt.forallvars)
            feed_dict = {adt.forallvars[i].input_var: inputs[i] for i in range(len(inputs))}
            train_outs_losses = pdt.train_fn(feed_dict, sess)
            train_outs = train_outs_losses[0:ntrain_outs]
            losses = train_outs_losses[ntrain_outs:]
            print("epoch:", epoch, " of ", num_epochs, " i: ", i, "losses: ", losses)
            train_err += losses[-2]
            train_batches += 1
            gens = [next(gen) for gen in pdt.generators]
            if j % save_every == 0:
                print(dict(zip([axiom.name for axiom in adt.axioms], losses[0:-1])))
                save_path = os.path.join(save_dir, "model.ckpt")
                save_path = saver.save(sess, save_path)
                print("Model saved in file: %s" % save_path)
            j = j + 1
            #     # Savs statistics
            #     loss_sum = np.sum(losses)
            #     stats['loss_sums'].append(loss_sum)
            #     loss_var = np.var(losses)
            #     stats['loss_vars'].append(loss_var)
            #     stat_sfx = "epoch_%s_run_%s_stats" % (epoch, i)
            #     if compress:
            #         np.savez_compressed(stats_path, **stats)
            #     else:
            #         np.savez(stats_path, **stats)
            #     # Save Params
            #     sfx2 = "epoch_%s_run_%sloss_%s" % (epoch, i, str(loss_sum))
            #     path = os.path.join(save_dir, sfx2)
            #     # adt.save_params(path, compress=compress)
        print("epoch: ", epoch, " Total loss per epoch: ", train_err)
    # path = os.path.join(save_dir, "final" + sfx)
    # adt.save_params(path)
