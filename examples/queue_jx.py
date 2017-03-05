from mnist import *
# from ig.util import *
from pdt.train_tf import *
from pdt.common import *
from pdt.util.misc import *
from pdt.util.io import mk_dir
from pdt.util.generators import infinite_samples, infinite_batches
from pdt.types import *
from common import handle_options, load_train_save


def queue_adt(train_data,
              options,
              queue_shape=(28, 28, 1),
              push_args={},
              pop_args={},
              empty_queue_args={},
              item_shape=(28, 28, 1),
              batch_size=512,
              nitems=3):
    """Construct a queue abstract data type"""
    # Types - a Queue of Item
   # Queue = Type(queue_shape, 'Queue')
    Queue = Type (queue_shape, 'Queue')
    Item = Type(item_shape, 'Item')

    # Interface

    # Push an Item onto a queue to create a new queue
    push = Interface([Queue, Item], [Queue], 'push', **push_args) 
    #push = Interface([Queue, Item], [Queue], 'push', **push_args)
    
    # Pop an Item from a queue, returning a new queue and the item
    pop = Interface([Queue], [Queue, Item], 'pop', **pop_args)
    funcs = [push, pop]

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
    '''
    When push N items onto queue, then pop N item off a queue, 
        want to get the N items in the same order that you pushed them.

    '''
    axioms = []
    queue = empty_queue.batch_input_var
    queues = [queue]
    for i in range(nitems):
        (queue,) = push(queue, items[i].input_var) # pushed the item onto the queue
        queues.append(queue)
        pop_queue = queue
        for j in range(i+1):
            (pop_queue, pop_item) = pop(pop_queue) # when you pop item from queue
            axiom = Axiom((pop_item,), (items[j].input_var,), 'item-eq%s-%s' %(i, j))
            axioms.append(axiom)

            
    train_fn, call_fns = compile_fns(funcs, consts, forallvars, axioms,
                                     train_outs, options)
    queue_adt = AbstractDataType(funcs, consts, forallvars, axioms,
                                 name='queue')
    queue_pdt = ProbDataType(queue_adt, train_fn, call_fns,
                             generators, gen_to_inputs, train_outs)
    return queue_adt, queue_pdt


def queue_unqueue(n, queue, offset=0):
    lb = 0 + offset
    ub = 1 + offset
    imgs = []
    queues = []
    queues.append(queue)
    for i in range(n):
        new_img = floatX(X_train[lb+i:ub+i])
        imgs.append(new_img)
        (queue,) = push(queue,new_img)
        queues.append(queue)

    for i in range(n):
        (queue, old_img) = pop(queue)
        queues.append(queue)
        imgs.append(old_img)

    return queues + imgs

def mnistshow(x):
    plt.imshow(x.reshape(28, 28))

def internal_plot(images, push, pop, empty):
    queue = empty
    for i in range(len(images)):
        mnistshow(queue)
        plt.figure()
        (queue,) = push(queue, images[i])
    mnistshow(queue)

def main(argv):
    global adt, pdt, sess, X_train, sfx
    options = handle_options('queue', argv)

    mnist_data = load_dataset()
    X_train = mnist_data[0].reshape(-1, 28, 28, 1)
    #sfx = gen_sfx_key(('adt', 'nblocks', 'block_size'), options)
    sfx = gen_sfx_key(('adt', 'nitems'), options)

    empty_queue_args = {'initializer': tf.random_uniform_initializer}
    adt, pdt = queue_adt(X_train,
                         options,
                         push_args=options,
                         nitems=options['nitems'],
                         pop_args=options,
                         empty_queue_args=empty_queue_args,
                         batch_size=options['batch_size'])

    save_dir = mk_dir(sfx)
    sess = load_train_save(options, adt, pdt, sfx, save_dir)

if __name__ == "__main__":
    main(sys.argv[1:])
