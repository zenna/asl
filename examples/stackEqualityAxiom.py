from mnist import *
# from ig.util import *
from pdt.train_tf import *
from pdt.common import *
from pdt.util.misc import *
from pdt.util.io import mk_dir
from pdt.util.generators import infinite_samples, infinite_batches
from pdt.types import *
from common import handle_options, load_train_save


def eqstack_adt(train_data,
              options,
              eqstack_shape=(28, 28, 1),
              push_args={},
              pop_args={},
              empty_eqstack_args={},
              item_shape=(28, 28, 1),
              batch_size=512,
              nitems=3):
    """Construct a eqstack abstract data type"""
    # Types - a Eqstack of Item
    Eqstack = Type(eqstack_shape, 'Eqstack')
    Item = Type(item_shape, 'Item')

    # Interface

    # Push an Item onto a Eqstack to create a new eqstack
    push = Interface([Eqstack, Item], [Eqstack], 'push', **push_args)
    # Pop an Item from a eqstack, returning a new eqstack and the item
    pop = Interface([Eqstack], [Eqstack, Item], 'pop', **pop_args)
    funcs = [push, pop]

    # train_outs
    train_outs = []
    gen_to_inputs = identity

    # Consts
    # The empty eqstack is the eqstack with no items
    empty_eqstack = Const(Eqstack, 'empty_eqstack', batch_size, **empty_eqstack_args)
    consts = [empty_eqstack]

    # Vars
    # eqstack1 = ForAllVar(Eqstack)
    items = [ForAllVar(Item, str(i)) for i in range(nitems)]
    forallvars = items

    generators = [infinite_batches(train_data, batch_size, shuffle=True)
                  for i in range(nitems)]

    # Axioms
    axioms = []
    eqstack = empty_eqstack.batch_input_var
    eqstacks = [eqstack]
    for i in range(nitems):
        (eqstack,) = push(eqstack, items[i].input_var)
        eqstacks.append(eqstack)
        pop_eqstack = eqstack
        for j in range(i, -1, -1):
            (pop_eqstack, pop_item) = pop(pop_eqstack)
            axiom = Axiom((pop_item,), (items[j].input_var,), 'item-eq%s-%s' %(i, j))
            axioms.append(axiom)

            # Eqstack equivalence
            axiom = Axiom((pop_eqstack,), (eqstacks[j],), 'eqstack-eq%s-%s' %(i, j))
            axioms.append(axiom)
    train_fn, call_fns = compile_fns(funcs, consts, forallvars, axioms,
                                     train_outs, options)
    eqstack_adt = AbstractDataType(funcs, consts, forallvars, axioms,
                                 name='eqstack')
    eqstack_pdt = ProbDataType(eqstack_adt, train_fn, call_fns,
                             generators, gen_to_inputs, train_outs)
    return eqstack_adt, eqstack_pdt


def eqstack_uneqstack(n, eqstack, offset=0):
    lb = 0 + offset
    ub = 1 + offset
    imgs = []
    eqstacks = []
    eqstacks.append(eqstack)
    for i in range(n):
        new_img = floatX(X_train[lb+i:ub+i])
        imgs.append(new_img)
        (eqstack,) = push(eqstack,new_img)
        eqstacks.append(eqstack)

    for i in range(n):
        (eqstack, old_img) = pop(eqstack)
        eqstacks.append(eqstack)
        imgs.append(old_img)

    return eqstacks + imgs

def mnistshow(x):
    plt.imshow(x.reshape(28, 28))

def internal_plot(images, push, pop, empty):
    eqstack = empty
    for i in range(len(images)):
        mnistshow(eqstack)
        plt.figure()
        (eqstack,) = push(eqstack, images[i])
    mnistshow(eqstack)

def main(argv):
    global adt, pdt, sess, X_train, sfx
    options = handle_options('eqstack', argv)

    mnist_data = load_dataset()
    X_train = mnist_data[0].reshape(-1, 28, 28, 1)
    sfx = gen_sfx_key(('adt', 'nblocks', 'block_size'), options)

    empty_eqstack_args = {'initializer': tf.random_uniform_initializer}
    adt, pdt = eqstack_adt(X_train,
                         options,
                         push_args=options,
                         nitems=options['nitems'],
                         pop_args=options,
                         empty_eqstack_args=empty_eqstack_args,
                         batch_size=options['batch_size'])

    save_dir = mk_dir(sfx)
    sess = load_train_save(options, adt, pdt, sfx, save_dir)

if __name__ == "__main__":
    main(sys.argv[1:])
