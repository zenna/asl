from mnist import *
# from ig.util import *
from pdt.train_tf import *
from pdt.common import *
from pdt.util.misc import *
from pdt.util.io import mk_dir
from pdt.util.generators import infinite_samples, infinite_batches
from pdt.types import *
from common import handle_options, load_train_save


def stack_adt(train_data,
              options,
              stack_shape=(28, 28, 1),
              push_args={},
              pop_args={},
              empty_stack_args={},
              item_shape=(28, 28, 1),
              batch_size=512,
              nitems=3):
    """Construct a stack abstract data type"""
    # Types - a Stack of Item
    Stack = Type(stack_shape, 'Stack')
    Item = Type(item_shape, 'Item')

    # Interface

    # Push an Item onto a Stack to create a new stack
    push = Interface([Stack, Item], [Stack], 'push', **push_args)
    # Pop an Item from a stack, returning a new stack and the item
    pop = Interface([Stack], [Stack, Item], 'pop', **pop_args)
    funcs = [push, pop]

    # train_outs
    train_outs = []
    gen_to_inputs = identity

    # Consts
    # The empty stack is the stack with no items
    empty_stack = Const(Stack, 'empty_stack', batch_size, **empty_stack_args)
    consts = [empty_stack]

    # Vars
    # stack1 = ForAllVar(Stack)
    items = [ForAllVar(Item, str(i)) for i in range(nitems)]
    forallvars = items

    generators = [infinite_batches(train_data, batch_size, shuffle=True)
                  for i in range(nitems)]

    # Axioms
    axioms = []
    stack = empty_stack.batch_input_var
    stacks = [stack]
    for i in range(nitems):
        (stack,) = push(stack, items[i].input_var)
        stacks.append(stack)
        pop_stack = stack
        for j in range(i, -1, -1):
            (pop_stack, pop_item) = pop(pop_stack)
            axiom = Axiom((pop_item,), (items[j].input_var,), 'item-eq%s-%s' %(i, j))
            axioms.append(axiom)

            # Stack equivalence
            # axiom = Axiom((pop_stack,), (stacks[j],), 'stack-eq%s-%s' %(i, j))
            # axioms.append(axiom)
    train_fn, call_fns = compile_fns(funcs, consts, forallvars, axioms,
                                     train_outs, options)
    stack_adt = AbstractDataType(funcs, consts, forallvars, axioms,
                                 name='stack')
    stack_pdt = ProbDataType(stack_adt, train_fn, call_fns,
                             generators, gen_to_inputs, train_outs)
    return stack_adt, stack_pdt


def stack_unstack(n, stack, offset=0):
    lb = 0 + offset
    ub = 1 + offset
    imgs = []
    stacks = []
    stacks.append(stack)
    for i in range(n):
        new_img = floatX(X_train[lb+i:ub+i])
        imgs.append(new_img)
        (stack,) = push(stack,new_img)
        stacks.append(stack)

    for i in range(n):
        (stack, old_img) = pop(stack)
        stacks.append(stack)
        imgs.append(old_img)

    return stacks + imgs

def mnistshow(x):
    plt.imshow(x.reshape(28, 28))

def main(argv):
    global adt, pdt, sess, X_train, sfx
    options = handle_options('stack', argv)

    mnist_data = load_dataset()
    X_train = mnist_data[0].reshape(-1, 28, 28, 1)
    sfx = gen_sfx_key(('adt', 'nblocks', 'block_size'), options)

    empty_stack_args = {'initializer': tf.random_uniform_initializer}
    adt, pdt = stack_adt(X_train,
                         options,
                         push_args=options,
                         nitems=options['nitems'],
                         pop_args=options,
                         empty_stack_args=empty_stack_args,
                         batch_size=options['batch_size'])

    save_dir = mk_dir(sfx)
    sess = load_train_save(options, adt, pdt, sfx, save_dir)

if __name__ == "__main__":
    main(sys.argv[1:])
