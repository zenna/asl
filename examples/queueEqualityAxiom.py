from mnist import *
from pdt.train_tf import *
from pdt.types import *
from pdt.common import *
from wacacore.util.misc import *
from wacacore.util.io import mk_dir
from wacacore.util.generators import infinite_samples, infinite_batches
from common import handle_options


def eqqueue_adt(train_data,
              options,
              eqqueue_shape=(28, 28, 1),
              push_args={},
              pop_args={},
              empty_eqqueue_args={},
              item_shape=(28, 28, 1),
              batch_size=512,
              nitems=3):
    """Construct a eqqueue abstract data type"""
    # Types - a Eqqueue of Item
   # Eqqueue = Type(eqqueue_shape, 'Eqqueue')
    Eqqueue = Type (eqqueue_shape, 'Eqqueue')
    Item = Type(item_shape, 'Item')

    # Interface

    # Push an Item onto a eqqueue to create a new eqqueue
    push = Interface([Eqqueue, Item], [Eqqueue], 'push', **push_args)
    #push = Interface([Eqqueue, Item], [Eqqueue], 'push', **push_args)

    # Pop an Item from a eqqueue, returning a new eqqueue and the item
    pop = Interface([Eqqueue], [Eqqueue, Item], 'pop', **pop_args)
    funcs = [push, pop]

    # train_outs
    train_outs = []
    gen_to_inputs = identity

    # Consts
    # The empty eqqueue is the eqqueue with no items
    empty_eqqueue = Const(Eqqueue, 'empty_eqqueue', batch_size, **empty_eqqueue_args)
    consts = [empty_eqqueue]

    # Vars
    # eqqueue1 = ForAllVar(Eqqueue)
    items = [ForAllVar(Item, str(i)) for i in range(nitems)]
    forallvars = items

    generators = [infinite_batches(train_data, batch_size, shuffle=True)
                  for i in range(nitems)]

    # Axioms
    '''
    When push N items onto eqqueue, then pop N item off a eqqueue,
        want to get the N items in the same order that you pushed them.

    '''
    axioms = []
    eqqueue = empty_eqqueue.batch_input_var
    eqqueues = [eqqueue]
    for i in range(nitems):
        orig_eqqueue = eqqueue
        (push_eqqueue,) = push(orig_eqqueue, items[i].input_var) # pushed the item onto the eqqueue
        eqqueues.append(push_eqqueue)
        pop_eqqueue = push_eqqueue

        for j in range(i+1):
            # Item equivalence
            (pop_eqqueue, pop_item) = pop(pop_eqqueue) # when you pop item from queue
            axiom = Axiom((pop_item,), (items[j].input_var,), 'item-eq%s-%s' %(i, j))
            axioms.append(axiom)
            
            # (pop_eqqueue, pop_item) = pop(pop_eqqueue) # when you pop item from eqqueue
            # axiom = Axiom((pop_item,), (items[i].input_var,), 'item-eq%s-%s' %(i, i))
            # axioms.append(axiom)

            # Eqqueue equivalence, Case 1: Orig queue was empty
            if i==j:
                axiom = Axiom((pop_eqqueue,), (empty_eqqueue.batch_input_var,), 'eqqueue-eq%s-%s' %(i, j))
                axioms.append(axiom)

            # Eqqueue equivalence, Case 2: Orig queue had items
            else:
                (test_pop_eqqueue, test_pop_item) = pop(orig_eqqueue)
                (test_push_eqqueue, ) = push(test_pop_eqqueue, items[i].input_var)
                axiom = Axiom((pop_eqqueue,), (test_push_eqqueue,), 'eqqueue-eq%s-%s' %(i, j)) #queue.push(i)[0].pop()[0] == queue.pop()[0].push(i)[0]
                axioms.append(axiom)

        # Set next queue to support one more item
        eqqueue=push_eqqueue

    #FIXME: Remove train_fn and call_fns from datastructure
    train_fn, call_fns = None, None
    eqqueue_adt = AbstractDataType(funcs, consts, forallvars, axioms,
                                 name='eqqueue')
    eqqueue_pdt = ProbDataType(eqqueue_adt, train_fn, call_fns,
                             generators, gen_to_inputs, train_outs)
    return eqqueue_adt, eqqueue_pdt


def eqqueue_uneqqueue(n, eqqueue, offset=0):
    lb = 0 + offset
    ub = 1 + offset
    imgs = []
    eqqueues = []
    eqqueues.append(eqqueue)
    for i in range(n):
        new_img = floatX(X_train[lb+i:ub+i])
        imgs.append(new_img)
        (eqqueue,) = push(eqqueue,new_img)
        eqqueues.append(eqqueue)

    for i in range(n):
        (eqqueue, old_img) = pop(eqqueue)
        eqqueues.append(eqqueue)
        imgs.append(old_img)

    return eqqueues + imgs

def mnistshow(x):
    plt.imshow(x.reshape(28, 28))

def internal_plot(images, push, pop, empty):
    eqqueue = empty
    for i in range(len(images)):
        mnistshow(eqqueue)
        plt.figure()
        (eqqueue,) = push(eqqueue, images[i])
    mnistshow(eqqueue)

def main(argv):
    global adt, pdt, sess, X_train, sfx
    options = handle_options('eqqueue', argv)

    mnist_data = load_dataset()
    X_train = mnist_data[0].reshape(-1, 28, 28, 1)
    #sfx = gen_sfx_key(('adt', 'nblocks', 'block_size'), options)
    sfx = gen_sfx_key(('adt', 'nitems'), options)

    empty_eqqueue_args = {'initializer': tf.random_uniform_initializer}
    adt, pdt = eqqueue_adt(X_train,
                         options,
                         push_args=options,
                         nitems=options['nitems'],
                         pop_args=options,
                         empty_eqqueue_args=empty_eqqueue_args,
                         batch_size=options['batch_size'])

    datadir = os.path.join(os.environ['DATADIR'], "pdt")
    save_dir = mk_dir(sfx, datadir=datadir)
    options['sfx'] = sfx
    sess = train(adt, pdt, options, save_dir, sfx)

if __name__ == "__main__":
    main(sys.argv[1:])
