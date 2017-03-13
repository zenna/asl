import sys
from mnist import *
from pdt.types import * 
from common import handle_options
from wacacore.util.io import *


def scalarField2D(train_data,
                  batch_size=512):
    # Types and function Interfaces
    Field = Type((16,16), "Field")
    Image = Type((28,28), "Image")
    encode = Interface([Image], [Field], "encode")
    decode = Interface([Field], [Image], "decode")

    # Variables for use
    image = ForAllVar(Image, "image")

    # Defining the ADT functions
    (field, ) = encode([image.input_var])
    (image_output, ) = decode([field])
    image_axiom = Axiom([image_output], [image.input_var], "image-equality")

    # Generates training batches
    generators = []
    gen = infinite_batches(X_train, batch_size)
    generators.append(gen)

    # Defining ADT
    funcs = [encode, decode]
    const = []
    forallvars = [image]
    axioms = [image_axiom]
    field_adt = AbstactDataType(funcs, const, forallvars, axioms, name='field')

    # Defining PDT
    train_fn = None
    call_fns = None
    gen_to_inputs = identity 
    train_outs = [] 
    field_pdt = ProbDataType(field_adt, train_fn, call_fns, generators, gen_to_inputs, train_outs)

    return field_adt, field_pdt


def main(argv):
    mnist_data = load_dataset()
    X_train = mnist_data[0].reshape(-1, 28, 28, 1)
    options = handle_options('field', argv)
    options['dirname'] = gen_sfx_key(('adt',), options)

    adt, pdt = scalarField2D(X_train, batch_size=options['batch_size'])
    sess = train(adt, pdt, options)

if __name__ == "__main__":
    main(sys.argv[1:])