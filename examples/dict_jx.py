from mnist import *
# from ig.util import *
from pdt.train_tf import *
from pdt.common import *
from pdt.util.misc import *
from pdt.util.io import mk_dir
from pdt.util.generators import infinite_samples, infinite_batches
from pdt.types import *
from common import handle_options, load_train_save


def dict_adt(train_data,
              options,
              dict_shape=(28, 28, 1),
              push_args={},
              pop_args={},
              empty_dict_args={},
              item_shape=(28, 28, 1),
              batch_size=512,
              nitems=3):
    """Construct a dict abstract data type"""
    # Types - a Dict of Item
    Dict = Type (dict_shape, 'Dict')
    Item = Type(item_shape, 'Item')

    # Interface
    #TODO: Change / define functions

    # Insert Key / Value into Dict
    insert = Interface([Dict, Key, Value], [Dict], 'insert', **insert_args)

    # Search Value of Key in Dict
    search = Interface([Dict, Key], [Value], 'search', **search_args)

    # Remove Key from Dict
    remove = Interface([Dict, Key], [Dict], 'remove', **remove_args)

    funcs = [insert, search, remove]

    # train_outs
    train_outs = []
    gen_to_inputs = identity

    # Consts
    # The empty dict is the dict with no items
    empty_dict = Const(Dict, 'empty_dict', batch_size, **empty_dict_args)
    consts = [empty_dict]

    # Vars
    # dict1 = ForAllVar(Dict)
    items = [ForAllVar(Item, str(i)) for i in range(nitems)]
    forallvars = items

    generators = [infinite_batches(train_data, batch_size, shuffle=True)
                  for i in range(nitems)]

    # Axioms
    # TODO: Define the Axioms for dictionaries
    # TODO: Define how to train the dictionary
    '''
    Axioms
    1. When insert (key, value) and then search for value associated with key, should get the same value as what was inserted
    1.5. When insert new value for a given key, then should get the last value that was inserted
    2. When search for (key, value) that doesn't exist, then dictionary stays the same, return nothing
    3. When remove (key, value), when search for (key, value), return nothing

    Training
    - Notes: each image is unique and can be a unique dictionary
    - Could take every Nth element and use as a key, then add the next (n-1) images as values of the key
        - Check that the new values get added then returned fine
    - Remove M keys from the 

    '''
    axioms = []
    dict = empty_dict.batch_input_var
    dicts = [dict]
    for i in range(nitems):
        # (dict,) = push(dict, items[i].input_var) # pushed the item onto the dict
        dicts.append(dict)
        # pop_dict = dict
        
        # for j in range(i+1):
        #     (pop_dict, pop_item) = pop(pop_dict) # when you pop item from dict
        #     axiom = Axiom((pop_item,), (items[j].input_var,), 'item-eq%s-%s' %(i, j))
            # axioms.append(axiom)

            
    train_fn, call_fns = compile_fns(funcs, consts, forallvars, axioms,
                                     train_outs, options)
    dict_adt = AbstractDataType(funcs, consts, forallvars, axioms,
                                 name='dict')
    dict_pdt = ProbDataType(dict_adt, train_fn, call_fns,
                             generators, gen_to_inputs, train_outs)
    return dict_adt, dict_pdt


# def dict_undict(n, dict, offset=0):
#     lb = 0 + offset
#     ub = 1 + offset
#     imgs = []
#     dicts = []
#     dicts.append(dict)
#     for i in range(n):
#         new_img = floatX(X_train[lb+i:ub+i])
#         imgs.append(new_img)
#         (dict,) = push(dict,new_img)
#         dicts.append(dict)

#     for i in range(n):
#         (dict, old_img) = pop(dict)
#         dicts.append(dict)
#         imgs.append(old_img)

#     return dicts + imgs

def mnistshow(x):
    plt.imshow(x.reshape(28, 28))

def internal_plot(images, push, pop, empty):
    dict = empty
    for i in range(len(images)):
        mnistshow(dict)
        plt.figure()
        (dict,) = push(dict, images[i])
    mnistshow(dict)

def main(argv):
    global adt, pdt, sess, X_train, sfx
    options = handle_options('dict', argv)

    mnist_data = load_dataset()
    X_train = mnist_data[0].reshape(-1, 28, 28, 1)
    #sfx = gen_sfx_key(('adt', 'nblocks', 'block_size'), options)
    sfx = gen_sfx_key(('adt', 'nitems'), options)

    empty_dict_args = {'initializer': tf.random_uniform_initializer}
    adt, pdt = dict_adt(X_train,
                         options,
                         push_args=options,
                         nitems=options['nitems'],
                         pop_args=options,
                         empty_dict_args=empty_dict_args,
                         batch_size=options['batch_size'])

    save_dir = mk_dir(sfx)
    sess = load_train_save(options, adt, pdt, sfx, save_dir)

if __name__ == "__main__":
    main(sys.argv[1:])
