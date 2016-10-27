from pdt import *
from mnist import *
# from ig.util import *
from pdt.train_tf import *
from pdt.common import *
from pdt.io import *
from pdt.types import *
from common import handle_options, load_train_save


def gen_map_adt(n_facts,
                n_cities,
                options,
                map_shape=(10,),
                dist_args={},
                add_dist_args={},
                empty_map_args={},
                batch_size=512, nitems=3):
    """Construct a map abstract data type"""
    # Types - a Stack of Item
    Map = Type(map_shape, 'Map')
    City = Type((n_cities,), 'City') # one hot encoding
    Number = Type((1,), 'Number')

    # Interface
    # assert that the distance between two cities is a number
    add_dist = Interface([Map, City, City, Number], [Map], 'add_dist',
                         **add_dist_args)

    # Return a plausible distance between two cities
    # FIXME: When I used wrong type interface the wrong types the error wasnt clear
    dist = Interface([Map, City, City], [Number], 'dist', **dist_args)

    # Pop an Item from a map, returning a new map and the item
    funcs = [add_dist, dist]

    # train_outs
    train_outs = []

    ## Consts
    # The empty map is the map with no items
    empty_map = Const(Map, 'empty_map', batch_size, **empty_map_args)
    consts = [empty_map]

    # Vars
    # map1 = ForAllVar(Stack)
    forallvars = []
    cities = []
    distances = []
    for i in range(n_facts):
        city_i = ForAllVar(City, 'city_i_%s' % i)
        cities.append(city_i)
        forallvars.append(city_i)

        city_j = ForAllVar(City, 'city_j_%s' % i)
        cities.append(city_j)
        forallvars.append(city_j)

        dist_i_j = ForAllVar(Number, 'number_i_j_%s' % i)
        distances.append(dist_i_j)
        forallvars.append(dist_i_j)

    # Generators
    def generate_map(shape):
        return np.random.rand(n_cities, 2)

    # Gerenates a two 2 map
    map_generator = infinite_samples(np.random.rand, batch_size, (n_cities, 2))
    generators = [map_generator]

    def to_one_hot(city_is, total_num, batch_size):
        city_index = list(range(batch_size))
        zero = np.zeros((batch_size, total_num))
        zero[city_index, city_is] = 1
        return zero

    def generate_city_pairs(city_map):
        # For every fact
        #   For every element of batch
        # cities = [one_hot(i, n_cities, batch_size) for i in range(n_cities)]
        go = []
        city_index = list(range(batch_size))
        for i in range(n_facts):
            city_is = np.random.randint(n_cities, size=batch_size)
            city_js = np.random.randint(n_cities, size=batch_size)
            a = city_i_pos = city_map[0][city_index, city_is]
            b = city_j_pos = city_map[0][city_index, city_js]

            sqr_dist = np.sum((a - b)**2, 1)
            dist = np.sqrt(sqr_dist)
            go.append(to_one_hot(city_is, n_cities, batch_size))
            go.append(to_one_hot(city_js, n_cities, batch_size))
            go.append(dist.reshape(-1, 1))


        return go

    gen_to_inputs = generate_city_pairs

    # Axioms
    axioms = []
    curr_map = empty_map.batch_input_var
    for i in range(n_facts):
        city_i, city_j = cities[i*2], cities[i*2+1]
        city_dist_i_j = distances[i].input_var
        (curr_map,) = add_dist(curr_map, city_i, city_j, city_dist_i_j)
        (city_dist,) = dist(curr_map, city_i, city_j)
        # import pdb; pdb.set_trace()

        axioms.append(Axiom((city_dist_i_j,), (city_dist,)))

    train_fn, call_fns = compile_fns(funcs, consts, forallvars, axioms,
                                     train_outs, options)
    map_adt = AbstractDataType(funcs, consts, forallvars, axioms,
                               name='map')
    map_pdt = ProbDataType(map_adt, train_fn, call_fns,
                           generators, gen_to_inputs, train_outs)
    return map_adt, map_pdt


def main(argv):
    options = handle_options('map', argv)
    sfx = gen_sfx_key(('adt', 'nblocks', 'block_size'), options)
    empty_map_args = {'initializer': tf.random_uniform_initializer}
    map_adt, map_pdt = gen_map_adt(1,
                                   10,
                                   options,
                                   dist_args=options,
                                   add_dist_args=options,
                                   empty_map_args=empty_map_args,
                                   batch_size=options['batch_size'])

    graph = tf.get_default_graph()
    writer = tf.train.SummaryWriter('/home/zenna/repos/pdt/pdt/log', graph)
    save_dir = mk_dir(sfx)
    load_train_save(options, map_adt, map_pdt, sfx, save_dir)
    push, pop = map_pdt.call_fns


if __name__ == "__main__":
    # ipython -- examples/map.py --template=res_net --nblocks=1 --block_size=1 -u adam -l 0.0001 --nitems=2 --batch_size=128 --train 1 --num_epochs=1000 --layer_width 36
    main(sys.argv[1:])
