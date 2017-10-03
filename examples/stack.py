from mnist import *
from pdt.train_tf import *
from wacacore.util.misc import *
from wacacore.util.io import mk_dir, handle_args, gen_sfx_key
from wacacore.util.generators import infinite_samples, infinite_batches
from pdt.types import *
from common import default_benchmark_options
from tensortemplates.util.io import add_additional_options


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
    interfaces = [push, pop]

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
    # train_fn, call_fns = compile_fns(interfaces, consts, forallvars, axioms,
                                    #  train_outs, options)
    train_fn = None
    call_fns = None
    stack_adt = AbstractDataType(interfaces, consts, forallvars, axioms,
                                 name='stack')
    stack_pdt = ProbDataType(stack_adt, train_fn, call_fns,
                             generators, gen_to_inputs, train_outs)
    return stack_adt, stack_pdt


def mse(a, b):
    return np.mean(np.square(a - b))


# Check for loss from one example of pushing and popping
def stack_unstack(nitems, push, pop, empty, items):
    stack = empty
    stacks = [stack]
    losses = []
    for i in range(nitems):
        (stack,) = push(stack, items[i])
        stacks.append(stack)
        pop_stack = stack
        for j in range(i, -1, -1):
            (pop_stack, pop_item) = pop(pop_stack)
            loss = mse(pop_item, items[j])
            losses.append(loss)
    return np.sum(losses), losses


def mnistshow(x):
    plt.imshow(x.reshape(28, 28))


def internal_plot(images, push, pop, empty):
    stack = empty
    for i in range(len(images)):
        mnistshow(stack)
        plt.figure()
        (stack,) = push(stack, images[i])
    mnistshow(stack)


def default_stack_options():
    """Default options for scalar field"""
    return {'field_shape': (eval, "(16, 16)"),
            'name': (str, 'Stack'),
            'nitems': (int, 3)}


def combine_options():
    """Get options by combining default and command line"""
    cust_options = {}
    argv = sys.argv[1:]

    # add default options like num_iterations
    cust_options.update(default_benchmark_options())

    # add render specific options like width/height
    cust_options.update(default_stack_options())

    # Update with options passed in through command line
    cust_options.update(add_additional_options(argv))

    options = handle_args(argv, cust_options)
    return options


def main(argv):
    global adt, pdt, sess, X_train, sfx
    options = combine_options()
    mnist_data = load_dataset()
    X_train = mnist_data[0].reshape(-1, 28, 28, 1)
    empty_stack_args = {'initializer': tf.random_uniform_initializer}
    adt, pdt = stack_adt(X_train,
                         options,
                         push_args=options,
                         nitems=options['nitems'],
                         pop_args=options,
                         empty_stack_args=empty_stack_args,
                         batch_size=options['batch_size'])

    options['dirname'] = gen_sfx_key(('adt', 'nitems'), options)
    sess = train(adt, pdt, options)


if __name__ == "__main__":
    main(sys.argv[1:])
