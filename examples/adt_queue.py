from mnist import load_dataset
from pdt.train_tf import compile_fns
from pdt.common import gen_sfx_key
from pdt.util.io import infinite_batches, identity, mk_dir
from pdt.types import (Type, Interface, Const, Axiom, ForAllVar,
                       AbstractDataType, ProbDataType)
from common import handle_options, load_train_save
import tensorflow as tf
import sys


def gen_queue_adt(train_data,
                  options,
                  queue_shape=(28, 28, 1),
                  enqueue_args={},
                  dequeue_args={},
                  empty_queue_args={},
                  item_shape=(28, 28, 1),
                  batch_size=512,
                  nitems=3):
    """Construct a queue abstract data type"""
    # Types - a Queue of Item
    Queue = Type(queue_shape, 'Queue')
    Item = Type(item_shape, 'Item')

    # Interface

    # Push an Item onto a Queue to create a new queue
    enqueue = Interface([Queue, Item], [Queue], 'enqueue', **enqueue_args)
    # Pop an Item from a queue, returning a new queue and the item
    dequeue = Interface([Queue], [Queue, Item], 'dequeue', **dequeue_args)
    funcs = [enqueue, dequeue]

    # train_outs
    train_outs = []
    gen_to_inputs = identity

    # Consts
    # The empty queue is the queue with no items
    empty_queue = Const(Queue, 'empty_queue', batch_size, **empty_queue_args)
    consts = [empty_queue]

    # Vars
    # queue1 = ForAllVar(Queue)
    items = [ForAllVar(Item, str(i)) for i in range(nitems)]
    forallvars = items

    generators = [infinite_batches(train_data, batch_size, shuffle=True)
                  for i in range(nitems)]

    # Axioms
    axioms = []
    queue = empty_queue.batch_input_var
    for i in range(nitems):
        (queue,) = enqueue(queue, items[i].input_var)
        dequeue_queue = queue
        for j in range(i + 1):
            (dequeue_queue, dequeue_item) = dequeue(dequeue_queue)
            axiom = Axiom((dequeue_item,), (items[j].input_var,))
            axioms.append(axiom)
    train_fn, call_fns = compile_fns(funcs, consts, forallvars, axioms,
                                     train_outs, options)
    queue_adt = AbstractDataType(funcs, consts, forallvars, axioms,
                                 name='queue')
    queue_pdt = ProbDataType(queue_adt, train_fn, call_fns,
                             generators, gen_to_inputs, train_outs)
    return queue_adt, queue_pdt


def main(argv):
    options = handle_options('queue', argv)

    mnist_data = load_dataset()
    X_train = mnist_data[0].reshape(-1, 28, 28, 1)
    sfx = gen_sfx_key(('adt', 'nblocks', 'block_size'), options)

    empty_queue_args = {'initializer': tf.random_uniform_initializer}
    queue_adt, queue_pdt = gen_queue_adt(X_train, options, enqueue_args=options,
                         nitems=options['nitems'], dequeue_args=options,
                         empty_queue_args=empty_queue_args,
                         batch_size=options['batch_size'])

    graph = tf.get_default_graph()
    save_dir = mk_dir(sfx)
    load_train_save(options, queue_adt, queue_pdt, sfx, save_dir)
    enqueue, dequeue = queue_pdt.call_fns


if __name__ == "__main__":
    main(sys.argv[1:])
