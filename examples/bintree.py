from mnist import *
# from ig.util import *
from pdt.train_tf import *
from pdt.common import *
from wacacore.util.misc import *
from wacacore.util.io import mk_dir
from wacacore.util.generators import infinite_samples, infinite_batches
from pdt.types import *
from common import handle_options, load_train_save


def gen_binary_tree_adt(train_data,
                        options,
                        binary_tree_shape=(28, 28, 1),
                        left_tree_args={},
                        right_tree_args={},
                        get_item_args={},
                        item_shape=(28*28,),
                        batch_size=512):
    """Construct a Binary Tree Data Structure"""
    BinTree = Type(binary_tree_shape. 'Binary_Tree')
    Item = Type(item_shape, 'Item')
    make = Interface([BinTree, Item, BinTree], [BinTree], 'make', **make_args)
    left_tree = Interface([BinTree], [BinTree], 'left_tree', left_tree_args)
    right_tree = Interface([BinTree], [BinTree], 'right_tree', right_tree_args)
    get_item = Interface([BinTree], [Item], 'get_item', get_item_args)
    # is_empty = Interface([BinTree], [BoolType])

    # Vars
    bintree1 = ForAllVar(BinTree)
    bintree2 = ForAllVar(BinTree)
    item1 = ForAllVar(Item)
    # error = Constant(np.random.rand(item_shape))

    # axiom1 = Axiom(left_tree(create), error)
    make_stuff = make(bintree1.input_var, item1.input_var, bintree2.input_var)
    axiom2 = Axiom(left_tree(*make_stuff), (bintree1.input_var,))
    # axiom3 = Axiom(right_tree(create), error)
    axiom4 = Axiom(right_tree(*make_stuff), (bintree2.input_var,))
    # axiom5 = Axiom(item(create), error) # FIXME, how to handle True
    axiom6 = Axiom(get_item(*make_stuff), (item1.input_var,))
    # axiom7 = Axiom(is_empty(create), True)
    # axiom8 = Axiom(is_empty(make(bintree1.input_var, item1, bintree2)), False)
    interfaces = [make, left_tree, right_tree, get_item]
    # axioms = [axiom1, axiom2, axiom3, axiom4, axiom5, axiom6, axiom6, axiom7. axiom8]
    axioms = [axiom2, axiom4, axiom6]
    forallvars = [bintree1, bintree2, item1]
    generators = [infinite_samples(np.random.rand, batch_size, binary_tree_shape),
                  infinite_samples(np.random.rand, batch_size, binary_tree_shape),
                  infinite_minibatches(train_data, batch_size, True)]

    train_fn, call_fns = compile_fns(funcs, consts, forallvars, axioms,
                                     train_outs, options)
    binary_tree_adt = AbstractDataType(funcs, consts, forallvars, axioms,
                                       name='binary_tree')
    binary_tree_pdt = ProbDataType(binary_tree_adt, train_fn, call_fns,
                                   generators, gen_to_inputs, train_outs)
    return binary_tree_adt, binary_tree_pdt


def main(argv):
    global adt, pdt, sess
    options = handle_options('number', argv)
    sfx = gen_sfx_key(('adt', 'template', 'nblocks', 'block_size'), options)
    zero_args = {'initializer': tf.random_uniform_initializer}


    adt, pdt = gen_binary_tree_adt(options,
                                   number_shape=(5,),
                                   succ_args=options,
                                   add_args=options,
                                   mul_args=options,
                                   encode_args=options,
                                   decode_args=options,
                                   zero_args=zero_args,
                                   batch_size=options['batch_size'])

    save_dir = mk_dir(sfx)
    sess = load_train_save(options, adt, pdt, sfx, save_dir)

if __name__ == "__main__":
    main(sys.argv[1:])
    # ipython -- number.py -l 0.0001 -u adam --batch_size=512
