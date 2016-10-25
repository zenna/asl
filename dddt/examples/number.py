from dddt.types import *
from dddt.train_tf import *
from dddt.common import *

# theano.config.optimizer = 'None'
# theano.config.optimizer = 'fast_compile'


def number_adt(options, niters=3, number_shape=(5,), batch_size=64,
               succ_args={}, add_args={}, mul_args={},
               encode_args={}, decode_args={}, zero_args={}):
    # Types
    Number = Type(number_shape, "number")
    BinInteger = Type((1,), "bin_integer")  # A python integer

    # Interface
    succ = Interface([Number], [Number], 'succ', **succ_args)
    # add = Interface([Number, Number], [Number], 'add', **add_args)
    # mul = Interface([Number, Number], [Number], 'mul', **mul_args)
    encode = Interface([BinInteger], [Number], 'encode', **encode_args)
    decode = Interface([Number], [BinInteger], 'decode', **decode_args)
    funcs = [succ, encode, decode]
    # funcs = [succ, add, mul, encode, decode]

    # Vars
    # a = ForAllVar(Number)
    # b = ForAllVar(Number)
    bi = ForAllVar(BinInteger, "bi")
    # bj = ForAllVar(BinInteger)

    # forallvars = [bi, bj]
    forallvars = [bi]

    # Consts
    zero = Const(Number, "zero", batch_size, **zero_args)
    zero_batch = zero.batch_input_var
    consts = [zero]
    # consts = []

    # axioms
    # biv = tf.Print(bi.input_var, [bi.input_var], message="buv!")
    biv = bi.input_var
    (encoded1,) = encode(biv)
    # encoded1 = tf.Print(encoded1, [encoded1], message="message!")
    # (encoded2,) = encode(bj)

    axiom_zero = Axiom(decode(zero_batch), (0.0,))

    axiom_ed = Axiom(decode(encoded1), (biv,))
    (succ_encoded,) = succ(encoded1)
    axiom_succ_ed = Axiom(decode(succ_encoded), (bi.input_var + 1,))
    #
    axioms = [axiom_ed, axiom_succ_ed]
    #
    # a = encoded1
    # b = encoded2
    #
    # (succ_b,) = succ(b)
    # mul_a_succ_b = mul(a, succ_b)
    # mul_axiom2_rhs = mul(a, b) + [a]
    #
    # add_axiom1 = Axiom(add(a, zero_batch), (a,))
    # add_axiom2 = Axiom(add(a, succ_b), succ(*add(a, b)))
    # mul_axiom1 = Axiom(mul(a, zero_batch), (zero_batch,))
    # mul_axiom2 = Axiom(mul(a, succ_b), add(*mul_axiom2_rhs))
    # arith_axioms = [add_axiom1, add_axiom2, mul_axiom1, mul_axiom2]
    # axioms = encode_axioms + arith_axioms
    # axioms = [axiom_ed]

    # generators
    def realistic_nums(*shape):
        q = np.array(np.random.randint(0, 10, shape), dtype='float32')
        # q = np.array(np.random.zipf(1.7, shape) +
        #                 np.random.randint(-1, 10, shape), dtype='float32')
        # # import pdb; pdb.set_trace()
        return q
    generators = [infinite_samples(realistic_nums, batch_size, (1,))
                  for i in range(1)]

    train_outs = []
    gen_to_inputs = identity

    train_fn, call_fns = compile_fns(funcs, consts, forallvars, axioms,
                                     train_outs, options)
    number_adt = AbstractDataType(funcs, consts, forallvars, axioms,
                                  name='natural_number')
    number_pdt = ProbDataType(number_adt, train_fn, call_fns,
                              generators, gen_to_inputs, train_outs)
    return number_adt, number_pdt

from common import handle_options, load_train_save

def main(argv):
    # Args
    global options
    global test_files, train_files
    global views, outputs, net
    global push, pop
    global X_train
    global adt, pdt
    global sfx
    global save_dir

    options = handle_options('number', argv)
    sfx = gen_sfx_key(('adt', 'template', 'nblocks', 'block_size'), options)
    zero_args = {'initializer': tf.random_uniform_initializer}


    adt, pdt = number_adt(options,
                          number_shape=(5,),
                          succ_args=options, add_args=options,
                          mul_args=options, encode_args=options,
                          decode_args=options,
                          zero_args=zero_args,
                          batch_size=options['batch_size'])

    save_dir = mk_dir(sfx)
    load_train_save(options, adt, pdt, sfx, save_dir)

if __name__ == "__main__":
    main(sys.argv[1:])
