from mnist import load_dataset
from pdt.train_tf import compile_fns
from pdt.common import gen_sfx_key
from pdt.io import infinite_batches, identity, mk_dir
from pdt.types import (Type, Interface, Const, Axiom, ForAllVar,
                       AbstractDataType, ProbDataType)
from common import handle_options, load_train_save
import tensorflow as tf
import sys
import os

def gen_set_adt(train_data,
                options,
                set_shape=(11,),
                item_shape=(28, 28, 1),
                store_args={},
                is_in_args={},
                size_args={},
                is_empty_args={},
                empty_set_args={},
                batch_size=512,
                nitems=3):
    # Types
    Set = Type(set_shape, 'Set')
    Item = Type(item_shape, 'Item')
    Integer = Type((1,), 'Integer')
    Bool = Type((1,), 'Bool')

    TRUE_NUM = 1.0
    FALSE_NUM = 0.0

    # Interface

    store = Interface([Set, Item], [Set], 'store', **store_args)
    is_in = Interface([Set, Item], [Bool], 'is_in', **is_in_args)
    size = Interface([Set], [Integer], 'size', **size_args)
    is_empty = Interface([Set], [Bool], 'is_empty', **is_empty_args)
    # union = Interface([Set, Item], [Set], 'push', **push_args)
    # difference = Interface([Set], [Set, Item], 'pop', **pop_args)
    # subset = Interface([Set, Set], [Boolean], 'pop', **pop_args)

    funcs = [store, is_in, is_empty, size]

    # train_outs
    train_outs = []
    gen_to_inputs = identity

    # Consts
    empty_set = Const(Set, 'empty_set', batch_size, **empty_set_args)
    consts = [empty_set]

    # Vars
    items = [ForAllVar(Item, 'item_%s' % i) for i in range(nitems)]
    forallvars = items

    empty_set_bv = empty_set.batch_input_var

    axioms = []
    # axiom_1 = Axiom(is_empty(empty_set_bv), (0.5,))
    # axioms.append(axiom_1)

    (non_empty_store, ) = store(empty_set_bv, items[0])
    (is_not_empty, ) = is_empty(non_empty_store)
    axiom_2 = Axiom((is_not_empty, ), (FALSE_NUM,))
    axioms.append(axiom_2)

    axiom_3 = Axiom(size(empty_set_bv), (0.0,))
    axioms.append(axiom_3)

    # axiom4 = Axiom(is_in(empty_set, item1), (TRUE_NUM,))
    # item1_in_set1 = is_in(store(set1, item1))
    # axiom5 = CondAxiom(i1, i2, item1_in_set1, (1,),
    #                            item1_in_set1, (is_in(set1, i1)))

    # # union axioms
    # axiom6 = Axiom(union(empty_set, set2), set2)
    # axiom7 = Axiom(union(store(set1, item1), set2),
    #                store(union(set1, set2), item1))
    #
    # # intersect axioms
    # axiom8 = Axiom(intersect(empty_set, set2), empty_set)
    # intersect_store = intersect(store(set1,), item1, set2)
    # axiom9 = CondAxiom(is_in(T, item1), (1,),
    #                    intersect_store, store(intersect(set1, set2), item1),
    #                    intersect_store, interect(set1, set2))

    # Generators
    generators = [infinite_batches(train_data, batch_size, shuffle=True)
                  for i in range(nitems)]
    train_fn, call_fns = compile_fns(funcs, consts, forallvars, axioms,
                                     train_outs, options)
    set_adt = AbstractDataType(funcs, consts, forallvars, axioms,
                                 name='set')
    set_pdt = ProbDataType(set_adt, train_fn, call_fns,
                             generators, gen_to_inputs, train_outs)
    return set_adt, set_pdt

def main(argv):

    options = handle_options('set', argv)

    mnist_data = load_dataset()
    X_train = mnist_data[0].reshape(-1, 28, 28, 1)
    sfx = gen_sfx_key(('adt', 'nblocks', 'block_size'), options)

    empty_set_args = {'initializer': tf.random_uniform_initializer}
    set_adt, set_pdt = gen_set_adt(X_train,
                                   options,
                                   store_args=options,
                                   is_in_args=options,
                                   size_args=options,
                                   is_empty_args=options,
                                   empty_set_args=empty_set_args,
                                   nitems=options['nitems'],
                                   batch_size=options['batch_size'])

    graph = tf.get_default_graph()
    save_dir = mk_dir(sfx)

    path_name = os.path.join(os.environ['DATADIR'], 'graphs', sfx, )
    tf.train.SummaryWriter(path_name, graph)

    load_train_save(options, set_adt, set_pdt, sfx, save_dir)
    push, pop = pdt.call_fns


if __name__ == "__main__":
    main(sys.argv[1:])
