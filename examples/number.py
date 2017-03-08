from pdt.train_tf import *
from pdt.common import *
from wacacore.util.misc import *
from wacacore.util.io import mk_dir
from wacacore.util.generators import infinite_samples, infinite_batches
from pdt.types import *
from common import handle_options, load_train_save
import sys

def gen_number_adt(options,
                   niters=3,
                   number_shape=(5,),
                   batch_size=64,
                   succ_args={},
                   add_args={},
                   mul_args={},
                   encode_args={},
                   decode_args={},
                   zero_args={}):
    # Types
    Number = Type(number_shape, "number")
    BinInteger = Type((1,), "bin_integer")  # A python integer

    # Interface
    funcs = []
    succ = Interface([Number], [Number], 'succ', **succ_args)
    funcs.append(succ)
    add = Interface([Number, Number], [Number], 'add', **add_args)
    funcs.append(add)
    # mul = Interface([Number, Number], [Number], 'mul', **mul_args)
    encode = Interface([BinInteger], [Number], 'encode', **encode_args)
    funcs.append(encode)
    decode = Interface([Number], [BinInteger], 'decode', **decode_args)
    funcs.append(decode)
    # funcs = [succ, encode, decode]
    # funcs = [succ, add, mul, encode, decode]

    # Vars
    # a = ForAllVar(Number)
    # b = ForAllVar(Number)
    bi = ForAllVar(BinInteger, "b_i")
    bj = ForAllVar(BinInteger, "b_j")

    # forallvars = [bi, bj]
    forallvars = [bi, bj]

    # Consts
    zero = Const(Number, "zero", batch_size, **zero_args)
    zero_batch = zero.batch_input_var
    consts = [zero]
    # consts = []

    # axioms
    # biv = tf.Print(bi.input_var, [bi.input_var], message="buv!")
    biv = bi.input_var
    bjv = bj.input_var
    (encoded1,) = encode(biv)
    (encoded2,) = encode(bjv)
    # encoded1 = tf.Print(encoded1, [encoded1], message="message!")
    # (encoded2,) = encode(bj)
    axioms = []

    axiom_zero = Axiom(decode(zero_batch), (0.0,))
    axioms.append(axiom_zero)

    axiom_ed = Axiom(decode(encoded1), (biv,))
    axioms.append(axiom_ed)

    (succ_encoded,) = succ(encoded1)
    axiom_succ_ed = Axiom(decode(succ_encoded), (bi.input_var + 1,))
    axioms.append(axiom_succ_ed)
    #
    #
    # a = encoded1
    # b = encoded2
    #
    # (succ_b,) = succ(b)
    # mul_a_succ_b = mul(a, succ_b)
    # mul_axiom2_rhs = mul(a, b) + [a]

    # n + 0 = n
    add_axiom1 = Axiom(add(encoded1, zero_batch), (encoded1,))
    axioms.append(add_axiom1)

    # a + succ(b) == succ(a + b)
    (succ_j, ) = succ(encoded2)
    add_axiom2 = Axiom(add(encoded1, succ_j), succ(*add(encoded1, encoded2)))
    axioms.append(add_axiom2)

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
                  for i in range(2)]

    train_outs = []
    gen_to_inputs = identity

    train_fn, call_fns = compile_fns(funcs, consts, forallvars, axioms,
                                     train_outs, options)
    number_adt = AbstractDataType(funcs, consts, forallvars, axioms,
                                  name='natural_number')
    number_pdt = ProbDataType(number_adt, train_fn, call_fns,
                              generators, gen_to_inputs, train_outs)
    return number_adt, number_pdt

# def save():
#     all_variables = tf.all_variables()
#     savers = [saver = tf.train.Saver()

def main(argv):
    global adt, pdt, sess
    options = handle_options('number', argv)
    sfx = gen_sfx_key(('adt', 'template', 'nblocks', 'block_size'), options)
    zero_args = {'initializer': tf.random_uniform_initializer}


    adt, pdt = gen_number_adt(options,
                              number_shape=(10,),
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
    interface = {f.name:f for f in adt.funcs}
    add = interface['add'].to_python_lambda(sess)
    succ = interface['succ'].to_python_lambda(sess)
    decode = interface['decode'].to_python_lambda(sess)
    encode = interface['encode'].to_python_lambda(sess)

    vecs = [encode([[i]])[0] for i in range(20)]

    zero = adt.consts[0].input_var.eval(sess)
    succ_vecs = [zero]
    for i in range(20 - 1):
        succ_ed = succ(succ_vecs[-1])
        succ_vecs.append(succ_ed[0])
